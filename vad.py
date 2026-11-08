"""轻量向量化 VAD：用能量 + 过零率 + 谱平坦度区分人声与瞬态噪声

设计目标：进门之前快速判断"这段音频里到底有没有人声"，
把纯静音 / UI 点击声 / 键盘声 / 白噪声挡在模型外面（这类输入会让模型陷入复读循环）。
整段 5 分钟音频的分析耗时约 10~20 ms。
"""
import numpy as np

SR = 16000
WIN = 400      # 25 ms
HOP = 160      # 10 ms


def _frames(x, win=WIN, hop=HOP):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def analyze(pcm, sr=SR, zcr_max=0.30, flat_max=0.45, rel_db=10.0,
            min_seg_frames=3, merge_gap_frames=12, abs_floor_db=-58.0):
    """返回 {speech_seconds, ratio, segments, noise_floor_db, speech_frames}"""
    pcm = np.asarray(pcm, dtype=np.float32)
    if sr != SR:
        raise ValueError(f"expect {SR} Hz")

    fr = _frames(pcm) * np.hanning(WIN).astype(np.float32)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    energy = 20 * np.log10(rms + 1e-12)

    spec = np.abs(np.fft.rfft(fr, axis=1)) + 1e-9
    flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    zcr = np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)).mean(axis=1)

    noise_floor = float(np.percentile(energy, 10))
    thr = max(abs_floor_db, noise_floor + rel_db)
    mask = (energy > thr) & (zcr < zcr_max) & (flat < flat_max)

    # 合并短间隙、去掉过短的孤立瞬态（点击声通常只有 1~2 帧）
    idx = np.flatnonzero(mask)
    segs = []
    if idx.size:
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev <= merge_gap_frames:
                prev = i
            else:
                segs.append((start, prev))
                start = prev = i
        segs.append((start, prev))
        segs = [(s, e) for s, e in segs if (e - s + 1) >= min_seg_frames]

    speech_frames = sum(e - s + 1 for s, e in segs)
    speech_sec = speech_frames * HOP / SR
    dur = len(pcm) / SR
    return {
        "speech_seconds": round(speech_sec, 3),
        "ratio": round(speech_sec / dur, 4) if dur else 0.0,
        "segments": [(round(s * HOP / SR, 2), round((e * HOP + WIN) / SR, 2)) for s, e in segs],
        "noise_floor_db": round(noise_floor, 1),
        "duration": round(dur, 3),
    }


def has_speech(pcm, min_speech=0.3, **kw):
    """是否存在足够长的人声（低于 min_speech 秒视为无人声）"""
    st = analyze(pcm, **kw)
    st["ok"] = st["speech_seconds"] >= min_speech
    return st["ok"], st

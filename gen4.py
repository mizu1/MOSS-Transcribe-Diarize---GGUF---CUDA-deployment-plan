"""生成 4 人会议对话音频（8 轮交替）"""
import asyncio
import os

import edge_tts
import numpy as np
import soundfile as sf

SR = 16000
OUT = "audio"
VOICES = {
    "A": "zh-CN-XiaoxiaoNeural",   # 女
    "B": "zh-CN-YunjianNeural",    # 男
    "C": "zh-CN-XiaoyiNeural",     # 女
    "D": "zh-CN-YunxiNeural",      # 男
}
LINES = [
    ("A", "各位好，今天的会议主要讨论第三季度的产品路线图。"),
    ("B", "技术团队这边有个风险，核心模块的重构还没完成。"),
    ("C", "市场部希望提前两周发布，以便赶上行业展会。"),
    ("D", "我觉得可以分批上线，先开放给百分之十的用户。"),
    ("A", "那就按灰度方案推进，下周一提交详细的排期表。"),
    ("B", "好的，我明天把评估结果发到群里。"),
    ("C", "宣传物料这边我来协调，最晚周五给出初稿。"),
    ("D", "那我负责监控指标和回滚预案。"),
]


async def main():
    clips = []
    for i, (spk, text) in enumerate(LINES):
        p = os.path.join(OUT, f"_g4_{i}.mp3")
        await edge_tts.Communicate(text, VOICES[spk]).save(p)
        d, sr = sf.read(p, dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        if sr != SR:
            import librosa
            d = librosa.resample(d, orig_sr=sr, target_sr=SR)
        clips += [d, np.zeros(int(0.5 * SR), dtype=np.float32)]
        os.remove(p)
    full = np.concatenate(clips)
    full = full / max(1e-6, np.abs(full).max()) * 0.9
    sf.write(os.path.join(OUT, "dialog4.wav"), full, SR, subtype="PCM_16")
    print(f"dialog4.wav {len(full)/SR:.2f}s")


asyncio.run(main())

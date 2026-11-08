"""MOSS-Transcribe-Diarize 常驻 HTTP 服务（ctypes 调用 moss-transcribe.dll，模型只加载一次）

启动: python server.py [--port 8099]
接口:
  GET  /health                     健康检查
  POST /v1/audio/transcriptions    上传音频（multipart 字段 file，或直接二进制体）
"""
import argparse
import ctypes
import io
import json
import os
import re
import threading
import time
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import vad as vad_mod

ROOT = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(ROOT, "mtd", "build-cuda-shared", "moss-transcribe.dll")
GGUF = os.path.join(ROOT, "models", "gguf", "moss-transcribe-q4_0.gguf")
SR = 16000
LOG_TEXT = True          # 是否打印转写正文（--no-text 关闭）
LOG_LOCK = threading.Lock()

# ---- 进模型前的门禁 ----
VAD_ON = True            # --no-vad 关闭
VAD_MIN_SPEECH = 0.3     # 人声累计低于该秒数判定为"无人声"，直接跳过模型
VAD_TRIM = True          # --no-trim 关闭：裁掉首尾静音/噪声
VAD_PAD = 0.25           # 裁剪时两侧保留的缓冲
LOOP_GUARD = True        # --no-loop-guard 关闭：输出复读折叠


def log(msg):
    with LOG_LOCK:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


os.environ.setdefault("MTD_DEVICE", "cuda")

# DLL 依赖 cudart64_12.dll / cublas64_12.dll，把 CUDA 与构建目录加入搜索路径
for p in (
    os.path.join(ROOT, "mtd", "build-cuda-shared"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
):
    if os.path.isdir(p):
        os.add_dll_directory(p)

lib = ctypes.CDLL(DLL)
lib.moss_transcribe_capi_load.restype = ctypes.c_void_p
lib.moss_transcribe_capi_load.argtypes = [ctypes.c_char_p]
lib.moss_transcribe_capi_transcribe_pcm.restype = ctypes.c_void_p
lib.moss_transcribe_capi_transcribe_pcm.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.moss_transcribe_capi_free_string.argtypes = [ctypes.c_void_p]
lib.moss_transcribe_capi_last_error.restype = ctypes.c_char_p
lib.moss_transcribe_capi_last_error.argtypes = [ctypes.c_void_p]

LOCK = threading.Lock()
CTX = None
LOAD_SECONDS = 0.0


def load_model():
    global CTX, LOAD_SECONDS
    t0 = time.time()
    CTX = lib.moss_transcribe_capi_load(GGUF.encode())
    LOAD_SECONDS = time.time() - t0
    if not CTX:
        raise RuntimeError("模型加载失败")


SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.S)


def parse_segments(text):
    segs = []
    for m in SEG_RE.finditer(text):
        body = m.group(3).strip()
        if body:
            segs.append({
                "start": float(m.group(1)),
                "end": float(m.group(4)),
                "speaker": m.group(2),
                "text": body,
            })
    return segs


REPEAT_RE = re.compile(r"(.{1,40}?)\1{4,}", re.S)


def collapse_loops(text, max_cycle=4):
    """折叠模型在噪声/静音上产生的复读，例如 "Yeah. Yeah. Yeah. …" """
    if not LOOP_GUARD:
        return text, False
    hit = False
    for _ in range(6):
        m = REPEAT_RE.search(text)
        if not m:
            break
        text = text[: m.start()] + m.group(1) * min(max_cycle, 2) + "…"
        hit = True
    return text, hit


def decode_audio(raw):
    """支持 wav/mp3/flac 等，统一成 16k 单声道 float32"""
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        with wave.open(io.BytesIO(raw)) as w:
            n = w.getnframes()
            data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
            sr = w.getframerate()
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(data, dtype=np.float32)


def transcribe(pcm, max_new=0):
    n = len(pcm)
    ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    with LOCK:
        out = lib.moss_transcribe_capi_transcribe_pcm(CTX, ptr, n, SR, max_new)
        if not out:
            err = lib.moss_transcribe_capi_last_error(CTX)
            raise RuntimeError(err.decode() if err else "推理失败")
        try:
            text = ctypes.string_at(out).decode("utf-8", "replace")
        finally:
            lib.moss_transcribe_capi_free_string(out)
    return text


def extract_file(body, content_type):
    """兼容 multipart 与裸二进制"""
    if "multipart/form-data" in content_type:
        m = re.search(rb'boundary="?([^";\r\n]+)"?', content_type.encode())
        if not m:
            return body
        boundary = b"--" + m.group(1)
        for part in body.split(boundary):
            if b"\r\n\r\n" in part and b"filename=" in part:
                return part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
        return body
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_no_content(self):
        """被门禁滤掉：不返回任何内容体"""
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/health"):
            log(f"→ GET /health  来自 {self.client_address[0]}")
            self._send(200, {
                "status": "ok",
                "model": os.path.basename(GGUF),
                "backend": "CUDA",
                "load_seconds": round(LOAD_SECONDS, 2),
                "vad": {"enabled": VAD_ON, "min_speech": VAD_MIN_SPEECH, "trim": VAD_TRIM},
                "loop_guard": LOOP_GUARD,
            })
        else:
            log(f"→ GET {self.path}  404")
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/audio/transcriptions"):
            log(f"→ POST {self.path}  404")
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        log(f"→ POST {self.path}  来自 {self.client_address[0]}:{self.client_address[1]} "
            f"| {length / 1024:.1f} KB | {ctype or 'unknown'}")
        body = self.rfile.read(length)
        t0 = time.time()
        try:
            raw = extract_file(body, ctype)
            pcm = decode_audio(raw)
            full_dur = len(pcm) / SR

            # ---- 门禁：没有人声就不进模型，避免复读循环 ----
            vinfo = {"enabled": VAD_ON, "speech_seconds": None, "ratio": None}
            if VAD_ON:
                st = vad_mod.analyze(pcm)
                vinfo.update({
                    "speech_seconds": st["speech_seconds"],
                    "ratio": st["ratio"],
                    "noise_floor_db": st["noise_floor_db"],
                })
                if st["speech_seconds"] < VAD_MIN_SPEECH:
                    cost = time.time() - t0
                    log(f"← 丢弃：{full_dur:.2f}s 音频但人声仅 {st['speech_seconds']:.2f}s "
                        f"({st['ratio'] * 100:.1f}%) | 未调用模型 | 耗时 {cost * 1000:.0f}ms "
                        f"| 返回 204 无内容")
                    self._send_no_content()
                    return

                if VAD_TRIM and st["segments"]:
                    s0 = max(0.0, st["segments"][0][0] - VAD_PAD)
                    s1 = min(full_dur, st["segments"][-1][1] + VAD_PAD)
                    if s0 > 0.05 or s1 < full_dur - 0.05:
                        pcm = pcm[int(s0 * SR): int(s1 * SR)]
                        vinfo["trimmed"] = [round(s0, 2), round(s1, 2)]
                        vinfo["removed"] = round(full_dur - (s1 - s0), 2)
                if "trimmed" in vinfo:
                    offset = vinfo["trimmed"][0]
                else:
                    offset = 0.0
            else:
                offset = 0.0

            dur = len(pcm) / SR
            # ---- 生成上限兜底：按剩余时长给足余量，避免失控 -->
            max_new = int(min(8192, max(128, dur * 16 + 128)))
            text = transcribe(pcm, max_new=max_new)
            text, collapsed = collapse_loops(text)
            vinfo["loop_collapsed"] = collapsed

            segs = parse_segments(text)
            if offset:
                for s in segs:
                    s["start"] = round(s["start"] + offset, 2)
                    s["end"] = round(s["end"] + offset, 2)

            cost = time.time() - t0
            spk = sorted({s["speaker"] for s in segs})
            speech_tag = (f"人声 {vinfo['speech_seconds']:.2f}s | "
                          if vinfo["speech_seconds"] is not None else "")
            log(f"← {full_dur:.2f}s 音频 | {speech_tag}"
                f"耗时 {cost:.2f}s | RTF {cost / full_dur:.3f} ({full_dur / cost:.1f}x 实时) "
                f"| {len(segs)} 段 | 说话人 {','.join(spk) or '-'}"
                + (f" | 已裁首尾 {vinfo.get('removed', 0)}s" if "trimmed" in vinfo else "")
                + (" | 复读已折叠" if collapsed else ""))
            if LOG_TEXT:
                for s in segs:
                    t = s["text"]
                    if len(t) > 120:
                        t = t[:120] + "…"
                    log(f"     [{s['start']:6.2f}-{s['end']:6.2f}] {s['speaker']} {t}")
                if not segs:
                    log(f"     原始输出: {text[:200] or '(空)'}")
            self._send(200, {
                "text": "".join(s["text"] for s in segs) or text,
                "segments": segs,
                "speakers": spk,
                "duration": round(full_dur, 2),
                "elapsed": round(cost, 3),
                "rtf": round(cost / full_dur, 4) if full_dur else None,
                "vad": vinfo,
            })
        except Exception as e:
            log(f"← 错误 {type(e).__name__}: {e}")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-text", action="store_true", help="只打印请求与耗时，不打印转写正文")
    ap.add_argument("--no-vad", action="store_true", help="关闭人声门禁（噪声也会进模型）")
    ap.add_argument("--vad-min-speech", type=float, default=0.3,
                    help="人声累计低于该秒数则跳过模型，默认 0.3")
    ap.add_argument("--no-trim", action="store_true", help="只做门禁，不裁首尾静音")
    ap.add_argument("--no-loop-guard", action="store_true", help="关闭输出复读折叠")
    a = ap.parse_args()

    LOG_TEXT = not a.no_text
    VAD_ON = not a.no_vad
    VAD_MIN_SPEECH = a.vad_min_speech
    VAD_TRIM = not a.no_trim
    LOOP_GUARD = not a.no_loop_guard

    log(f"加载模型 {os.path.basename(GGUF)}")
    load_model()
    log(f"就绪 | 加载 {LOAD_SECONDS:.2f}s | http://{a.host}:{a.port}")
    log(f"接口: GET /health | POST /v1/audio/transcriptions | 正文日志 {'开' if LOG_TEXT else '关'}")
    log(f"人声门禁 {('开(阈值 %.2fs)' % VAD_MIN_SPEECH) if VAD_ON else '关'} | "
        f"首尾裁剪 {'开' if VAD_TRIM else '关'} | 复读折叠 {'开' if LOOP_GUARD else '关'}")
    log("等待请求 ...")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("收到中断，关闭服务")
        srv.shutdown()

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(ROOT, "mtd", "build-cuda-shared", "moss-transcribe.dll")
GGUF = os.path.join(ROOT, "models", "gguf", "moss-transcribe-q4_0.gguf")
SR = 16000

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

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {
                "status": "ok",
                "model": os.path.basename(GGUF),
                "backend": "CUDA",
                "load_seconds": round(LOAD_SECONDS, 2),
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/audio/transcriptions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        t0 = time.time()
        try:
            raw = extract_file(body, self.headers.get("Content-Type", ""))
            pcm = decode_audio(raw)
            text = transcribe(pcm)
            segs = parse_segments(text)
            cost = time.time() - t0
            self._send(200, {
                "text": "".join(s["text"] for s in segs) or text,
                "segments": segs,
                "speakers": sorted({s["speaker"] for s in segs}),
                "duration": round(len(pcm) / SR, 2),
                "elapsed": round(cost, 3),
                "rtf": round(cost / (len(pcm) / SR), 4) if len(pcm) else None,
            })
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    print(f"加载模型 {os.path.basename(GGUF)} ...", flush=True)
    load_model()
    print(f"就绪，耗时 {LOAD_SECONDS:.2f}s | http://{a.host}:{a.port}", flush=True)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.serve_forever()

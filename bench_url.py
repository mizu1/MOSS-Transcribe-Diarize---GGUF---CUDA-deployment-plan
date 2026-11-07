"""常驻 HTTP 服务基准：1s/6s/12s 音频的端到端延迟、服务端耗时与显存"""
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

import pynvml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8099"
pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)


def used_mb():
    return pynvml.nvmlDeviceGetMemoryInfo(H).used / 1024**2


class Sampler(threading.Thread):
    def __init__(self, base):
        super().__init__(daemon=True)
        self.peak = base
        self.base = base
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, used_mb())
            time.sleep(0.02)


def post(wav_path):
    with open(wav_path, "rb") as f:
        body = f.read()
    req = urllib.request.Request(
        f"{URL}/v1/audio/transcriptions", data=body,
        headers={"Content-Type": "audio/wav"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def main():
    with urllib.request.urlopen(f"{URL}/health", timeout=10) as r:
        print("health:", json.loads(r.read()))
    base = used_mb()
    print(f"服务空闲显存: {base:.0f} MiB\n")

    rows = []
    for sec in (1, 6, 12):
        wav = os.path.join(ROOT, "audio", f"t{sec}s.wav")
        post(wav)  # 预热
        lat, srv, peaks = [], [], []
        resp = None
        for _ in range(5):
            s = Sampler(base)
            s.start()
            resp, wall = post(wav)
            s.stop.set()
            s.join()
            lat.append(wall)
            srv.append(resp["elapsed"])
            peaks.append(s.peak)
        rows.append((sec, statistics.median(lat), min(lat), statistics.median(srv), max(peaks)))
        print(f"{sec}s 音频: 中位延迟 {statistics.median(lat)*1000:.0f} ms | "
              f"服务端 {statistics.median(srv)*1000:.0f} ms | "
              f"显存峰值 {max(peaks):.0f} MiB | 说话人 {resp['speakers']}")
        print(f"    {resp['text'][:70]}")

    print("\n" + "=" * 88)
    print(f"{'音频':<8}{'HTTP 中位延迟':<16}{'HTTP 最快':<14}{'服务端耗时':<14}"
          f"{'RTF':<9}{'峰值显存':<12}{'实时倍速'}")
    print("-" * 88)
    for sec, med, mn, srv, peak in rows:
        print(f"{str(sec)+'s':<8}{med*1000:<16.0f}{mn*1000:<14.0f}{srv*1000:<14.0f}"
              f"{srv/sec:<9.3f}{peak:<12.0f}{sec/srv:.1f}x")
    print("=" * 88)


if __name__ == "__main__":
    main()

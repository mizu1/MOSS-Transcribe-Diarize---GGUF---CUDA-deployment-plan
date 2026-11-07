"""Release 版 CLI：每次新进程的端到端耗时（含启动+加载）"""
import os
import statistics
import subprocess
import sys
import time

import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = "build-cuda-shared"
EXE = os.path.join(ROOT, "mtd", BUILD, "moss-transcribe.exe")
GGUF = os.path.join(ROOT, "models", "gguf", "moss-transcribe-q4_0.gguf")
LOG = os.path.join(ROOT, "_cli.log")

env = os.environ.copy()
env["PATH"] = (os.path.join(ROOT, "mtd", BUILD) + os.pathsep
               + r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin" + os.pathsep
               + env["PATH"])

print(f"{'音频':<8}{'端到端中位':<14}{'最快':<10}{'加载开销':<12}")
print("-" * 44)
base = None
for sec in (0.2, 1, 6, 12):
    wav = os.path.join(ROOT, "audio", f"t{sec}s.wav")
    ts = []
    for _ in range(3):
        t0 = time.time()
        with open(LOG, "wb") as fh:
            subprocess.run([EXE, "transcribe", GGUF, wav], cwd=ROOT, env=env,
                           stdout=fh, stderr=subprocess.STDOUT, timeout=600)
        ts.append(time.time() - t0)
    med = statistics.median(ts)
    if sec == 0.2:
        base = med
        print(f"{'0.2s(空载)':<8}{med:<14.2f}{min(ts):<10.2f}{'-':<12}")
    else:
        print(f"{str(sec)+'s':<8}{med:<14.2f}{min(ts):<10.2f}{(med-base):<12.2f}")

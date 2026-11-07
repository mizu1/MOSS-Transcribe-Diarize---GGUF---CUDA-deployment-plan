@echo off
REM 用法: run_gguf.bat <wav>  —— 单次 CLI 推理（含启动+加载，约 1.2~1.7s）
set "WAV=%~1"
if "%WAV%"=="" set "WAV=audio\t12s.wav"
set "PATH=d:\ProgramProject\Moss-Asr\mtd\build-cuda-shared;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;%PATH%"
cd /d d:\ProgramProject\Moss-Asr
mtd\build-cuda-shared\moss-transcribe.exe transcribe models\gguf\moss-transcribe-q4_0.gguf "%WAV%"

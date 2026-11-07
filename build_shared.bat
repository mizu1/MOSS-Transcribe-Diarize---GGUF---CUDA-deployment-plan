@echo off
call "D:\ProgramSystem\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;%PATH%"
cd /d d:\ProgramProject\Moss-Asr\mtd
cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release -DMT_GGML_CUDA=ON -DMT_BUILD_CLI=ON -DMT_SHARED=ON -DCMAKE_CUDA_ARCHITECTURES=89 "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" "-DCMAKE_CXX_FLAGS=-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" > d:\ProgramProject\Moss-Asr\sh_cfg.log 2>&1
if errorlevel 1 (
  echo CONFIG_FAILED
  exit /b 1
)
cmake --build build-cuda-shared -j > d:\ProgramProject\Moss-Asr\sh_build.log 2>&1
echo BUILD_EXIT=%errorlevel%

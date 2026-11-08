# MOSS-Transcribe-Diarize · GGUF + CUDA 部署

在 Windows + NVIDIA 显卡上跑 [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)（0.9B）的
量化 GGUF 版本，一次推理同时输出**转写文本 + 说话人标签 + 时间戳**。

推理后端是 [moss-transcribe.cpp](https://github.com/mudler/moss-transcribe.cpp)（ggml 的 C++17 移植），
不依赖 Python / PyTorch，模型、tokenizer、mel 参数全部打包在一个 GGUF 里。

本仓库提供的是**部署层**：Windows 构建脚本、常驻 HTTP 服务、基准脚本。上游源码未做任何修改。

---

## 能力

| 能力 | 说明 |
|---|---|
| 转写 | 中英文（本仓库仅验证中文） |
| 说话人分离 | 与转写联合生成，非后处理；实测 2 / 3 / 4 人均正确 |
| 重叠语音 | 同时说话的两段会各自输出，时间段重叠 |
| 时间戳 | 段级，秒为单位 |
| 长音频 | 单次最长 90 分钟（本仓库验证到 5 分钟） |

输出格式：

```
[0.16][S01]各位好，今天的会议主要讨论第三季度的产品路线图。[4.64]
[5.78][S02]技术团队这边有个风险，核心模块的重构还没完成。[9.88]
```

`[S01]` 等标签仅在本次音频内有效，不代表真实身份。

---

## 目录结构

```
Moss-Asr/
├─ models/gguf/           q4_0 GGUF 权重（约 510 MB，git 忽略）
├─ mtd/                   moss-transcribe.cpp 源码 + 构建产物（git 忽略）
├─ audio/                 测试音频（git 忽略）
├─ server.py              常驻 HTTP 服务（ctypes 调用 DLL，模型只加载一次）
├─ build_shared.bat       Windows 共享库编译脚本
├─ run_gguf.bat           单次 CLI 推理
├─ bench_url.py           HTTP 延迟 + 显存基准
├─ bench_cli_release.py   CLI 冷启动基准
└─ gen4.py                生成 4 人对话测试音频
```

---

## 前置条件

| 组件 | 版本 | 说明 |
|---|---|---|
| Windows | 10/11 x64 | |
| Visual Studio 2022 | C++ 桌面开发工作负载 | 提供 MSVC |
| CUDA Toolkit | **≥ 12.4**（推荐） | 低于 12.4 需额外绕参，见 FAQ |
| CMake | ≥ 3.18 | `pip install cmake` 即可 |
| Ninja | 任意 | `pip install ninja`；用于绕开 VS 的 CUDA 集成依赖 |
| Python | ≥ 3.10 | 仅服务层使用，推理本身不依赖 Python |
| NVIDIA 驱动 | 支持 CUDA 12.x | 显存 ≥ 4 GB（实测峰值 2.3 GB） |

---

## 部署

### 1. 取上游代码

```bash
git clone --depth 1 https://github.com/mudler/moss-transcribe.cpp.git mtd
cd mtd && git submodule update --init --recursive --depth 1
```

> 子模块是 ggml，必须拉取，否则无法编译。

### 2. 取模型

```bash
modelscope download --model mudler/moss-transcribe.cpp-gguf \
  moss-transcribe-q4_0.gguf --local_dir models/gguf
```

量化选择：`q4_0` 511 MB / `q5_0` 618 MB / `q8_0` 987 MB / `bf16` 1.83 GB。
上游验证 `q5_k` 及更高量化可保持 transcript 逐字节一致，`q4_0` 属最低档；精度优先选 `q5_0` 以上。

### 3. 编译（CUDA + 共享库）

```bat
build_shared.bat
```

脚本内容：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;%PATH%"
cd /d <repo>\mtd
cmake -B build-cuda-shared -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DMT_GGML_CUDA=ON -DMT_BUILD_CLI=ON -DMT_SHARED=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda-shared -j
```

要点：

- **必须 `-DCMAKE_BUILD_TYPE=Release`**，否则可能链到调试运行库，产物无法在普通进程里加载
- `CMAKE_CUDA_ARCHITECTURES` 按显卡算力填：RTX 40 系 = `89`，30 系 = `86`，20 系 = `75`
- `-DMT_SHARED=ON` 才会产出 `moss-transcribe.dll`（服务模式需要）
- 用 Ninja 而非 Visual Studio 生成器，可避免 `No CUDA toolset found`

### 4. 启动服务

```bash
python server.py --port 8099
```

启动时加载模型（约 0.5 s），之后常驻显存，不再重复加载。

控制台会按时间戳打印每一次请求与结果，便于确认"什么时候发了什么"：

```
[2026-11-08 08:08:43] 就绪 | 加载 0.46s | http://127.0.0.1:8099
[2026-11-08 08:08:43] 接口: GET /health | POST /v1/audio/transcriptions | 正文日志 开
[2026-11-08 08:08:43] 等待请求 ...
[2026-11-08 08:08:58] → POST /v1/audio/transcriptions  来自 127.0.0.1:63332 | 187.5 KB | audio/wav
[2026-11-08 08:08:59] ← 6.00s 音频 | 耗时 1.36s | RTF 0.226 (4.4x 实时) | 2 段 | 说话人 S01,S02
[2026-11-08 08:08:59]      [  0.16-  4.62] S01 各位好，今天的会议主要讨论第三季度的产品路线图。
[2026-11-08 08:08:59]      [  5.78-  6.02] S02 技术。
```

参数：

| 参数 | 说明 |
|---|---|
| `--port` | 监听端口，默认 8099 |
| `--host` | 监听地址，默认 127.0.0.1 |
| `--no-text` | 只打印请求与耗时，不打印转写正文（长音频日志更清爽） |
| `--no-vad` | 关闭人声门禁，所有音频都进模型 |
| `--vad-min-speech` | 人声累计低于该秒数就跳过模型，默认 `0.3` |
| `--no-trim` | 只做门禁，不裁掉首尾静音 |
| `--no-loop-guard` | 关闭输出复读折叠 |

日志内容包含：请求来源 IP/端口、上传体积、Content-Type、音频时长、推理耗时、RTF、段数、说话人列表，以及每段的时间戳与正文。

---

## 人声门禁（VAD）

模型在**几乎没有语音**的输入上会陷入复读，例如 0.2 秒静音会生成 15000 字符的 `Yeah. Yeah. Yeah. …`，
耗时从 1 秒膨胀到 70 秒。UI 点击声、键盘声、白噪声也会触发类似的失控。

服务在调用模型前加了三层防护：

**1. 门禁：没有足够人声就不进模型**

`vad.py` 用能量 + 过零率 + 谱平坦度做逐帧判定（25 ms 窗 / 10 ms 步长）：

| 特征 | 人声 | 瞬态噪声（点击/键盘） |
|---|---|---|
| 能量 | 持续高于底噪 | 只有 1~2 帧的尖峰 |
| 过零率 | 低（浊音）~中 | 高（宽频） |
| 谱平坦度 | 低（有共振峰结构） | 接近 1（接近白噪声） |

过滤掉短于 3 帧的孤立瞬态后，若人声累计不足 `--vad-min-speech` 秒，直接返回空结果并标记 `skipped: true`。
实测 5 分钟音频分析耗时约 217 ms（比 Silero VAD 快 14 倍）。

**2. 裁剪：去掉首尾的非语音段**，保留 0.25 秒缓冲，返回的时间戳自动补偿偏移。

**3. 兜底：按音频时长限制生成长度 + 折叠输出复读**，即使有漏网的噪声，最坏情况也被限制在秒级。

实测效果：

| 输入 | 传统方式 | 现在 |
|---|---|---|
| 0.2 s 静音 | 70.9 s，15340 字符复读 | **12 ms，直接跳过** |
| UI 点击声 6 s | 进模型 | **22 ms，跳过** |
| 键盘声 6 s | 进模型 | **31 ms，跳过** |
| 白噪声 4 s | 进模型 | **8 ms，跳过** |
| 1 s 人声 | 正常 | 正常，1.2 s |
| 语音+点击 41 s | 正常 | 正常，8 段 4 人 |

### 5. 调用

```bash
curl -X POST http://127.0.0.1:8099/v1/audio/transcriptions \
  -H "Content-Type: audio/wav" \
  --data-binary @audio/t12s.wav
```

也支持 multipart（字段名 `file`），以及 mp3 / flac 等格式，服务端自动转 16 kHz 单声道。

---

## 接口

### `GET /health`

```json
{"status":"ok","model":"moss-transcribe-q4_0.gguf","backend":"CUDA","load_seconds":0.55}
```

### `POST /v1/audio/transcriptions`

请求体：音频二进制（或 multipart `file` 字段）

响应：

```json
{
  "text": "各位好，今天的会议主要讨论第三季度的产品路线图。技术团队这边有个风险…",
  "segments": [
    {"start": 0.16, "end": 4.64, "speaker": "S01",
     "text": "各位好，今天的会议主要讨论第三季度的产品路线图。"},
    {"start": 5.78, "end": 9.88, "speaker": "S02",
     "text": "技术团队这边有个风险，核心模块的重构还没完成。"}
  ],
  "speakers": ["S01", "S02"],
  "duration": 12.0,
  "elapsed": 1.592,
  "rtf": 0.133
}
```

---

## 实测性能

RTX 4060 Laptop 8 GB，q4_0，Release，CUDA，常驻服务（热态）：

| 音频 | 端到端 | RTF | 实时倍速 |
|---:|---:|---:|---:|
| 1 s | 0.90 s | 0.96 | 1.0x |
| 6 s | 1.24 s | 0.21 | 4.8x |
| 12 s | 1.59 s | 0.13 | 7.5x |
| 91 s | 8.08 s | 0.089 | 11.2x |

- 显存：加载后常驻 ~0.9 GB，推理峰值 ~2.3 GB，**与音频长度基本无关**
- 冷启动（每次起新进程，含加载）：1 s / 6 s / 12 s 音频分别 1.23 / 1.48 / 1.68 s
- 同样 91 s 音频，PyTorch bf16 参考实现约 48.6 s（RTF 0.53），本方案约 8.1 s

**耗时构成**：编码器窗口固定 30 s（3000 mel 帧 / 480000 采样），所以 **≤30 s 的音频编码成本恒定**（本机约 0.75 s），
更长的音频按 30 s 切块；剩余时间花在逐 token 解码（约 15~50 tok/s），因此"输出多少字"比"音频多长"影响更大。

---

## 已知限制

- **非流式**：整段音频一次前向，不支持边说边出字
- **串行处理**：单 CUDA context 非线程安全，服务内用全局锁，并发请求会排队
  （实测 4 个 12 s 请求排队 5.08 s，吞吐约 9.4x 实时）
- **静音会复读**：纯静音或极短空白音频可能触发重复生成循环，做压测别用静音当基线
- **说话人编号不跨音频保持**：`S01` 仅本次请求内有效
- **Windows 服务化非上游方案**：`server.py` 是部署层代码，上游只提供 lib + CLI

---

## FAQ

**`CMake Error: No CUDA toolset found`**
VS 2022 未安装 CUDA 集成组件。改用 Ninja 生成器（见编译步骤），不要用 `-G "Visual Studio 17 2022"`。

**`fatal error C1189: unsupported Microsoft Visual Studio version`**
CUDA 版本低于 MSVC 要求。首选升级到 CUDA ≥ 12.4；若必须用旧版：

```bat
-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH" ^
-DCMAKE_CXX_FLAGS="-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
```

**`error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer`**
同上，MSVC 的 STL 硬性检查；`_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` 可绕过。

**`FileNotFoundError: Could not find module 'moss-transcribe.dll'`**
缺少 `cudart64_12.dll` / `cublas64_12.dll`，或 DLL 是 Debug 构建。检查项：
- `%CUDA_PATH%\bin` 在 `PATH` 里（`server.py` 已用 `os.add_dll_directory` 处理）
- 构建时带 `-DCMAKE_BUILD_TYPE=Release`

**`No module named 'mlx'`**
MLX 格式的模型（如 `mlx-*`）只能跑在 Apple Silicon，Windows/NVIDIA 无法使用，需要换成 GGUF 或 PyTorch 权重。

---

## 相关链接

- 上游模型：https://www.modelscope.cn/models/openmoss/MOSS-Transcribe-Diarize
- 上游代码：https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
- C++ 推理实现：https://github.com/mudler/moss-transcribe.cpp
- GGUF 权重：https://www.modelscope.cn/models/mudler/moss-transcribe.cpp-gguf

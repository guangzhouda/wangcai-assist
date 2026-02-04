# wangcai-assist

本项目是一个在 Windows 本地跑通的语音助手原型链路：

- KWS（唤醒词）：Picovoice Porcupine + PvRecorder
- VAD（断句/端点检测）：Silero VAD (ONNX)
- ASR（实时中英混合）：sherpa-onnx streaming zipformer
- LLM（对话）：DeepSeek（OpenAI 兼容 Chat Completions，支持 SSE 流式输出）
- TTS（语音合成）：
  - MeloTTS（默认；sherpa-onnx VITS onnx；支持中文/中英混合；CPU 友好）
  - CosyVoice2（本地模型；零样本固定音色）
  - OpenVoice V2（转音色：把底座 TTS 的音色转换成目标音色）
  - Matcha-TTS（预训练英文模型：Matcha + HiFiGAN，低显存/CPU 也可跑）

重要：本仓库 **不会提交任何模型文件/大文件/密钥**。模型请按 README 指引自行下载到本地 `model/`。

## 当前已完成

- `wakeword.py`：Porcupine 唤醒词监听（回调可扩展动作）
- `silero_vad.py`：Silero VAD ONNX 封装成可复用函数
- `sherpa_asr.py`：流式 ASR + VAD 切句 + pre-roll，支持部分结果单行刷新与 final 回调
- `llm_deepseek.py`：DeepSeek OpenAI 兼容客户端（`stream_chat()` / `chat()`）
- `voice_chat.py`：ASR final -> LLM stream -> **增量 TTS**（支持并行“合成/播放”流水线，降低卡顿）
- `main.py`：完整链路入口（KWS 唤醒 -> VAD/ASR -> LLM -> 增量 TTS，支持“休眠/退出”回到待机）
- `tts_melo.py` / `download_melo_model.py`：MeloTTS（中文/中英混合；默认 TTS）
- `tts_cosyvoice.py`：CosyVoice2 TTS（固定音色）
- `tts_openvoice.py`：OpenVoice V2 TTS（Melo 生成底座语音 + OpenVoice 转音色）
- `tts_compare.py`：一键对比不同 TTS 的生成耗时/RTF（输出到 `output/tts_compare/`）

## 后续计划（暂定）

- 唤醒链路打通：KWS -> VAD -> ASR -> LLM -> TTS（现在 voice_chat 默认 ASR 常开）
- 进一步优化低延迟：
  - 更细粒度的增量 TTS（更短首包、更平滑节奏）
  - onnxruntime-gpu / sherpa-onnx GPU 编译与加速（可选）
- 工具与扩展：
  - FC（Function Calling）对接本地工具能力
  - MCP（Model Context Protocol）接入外部能力/工具
  - 知识库（RAG）：本地文档/网页摘要/向量检索

## 目录与关键文件

- `main.py`：完整链路入口（带唤醒词）
- `wakeword.py`：唤醒词
- `silero_vad.py`：VAD
- `sherpa_asr.py`：实时 ASR（中英混合）
- `llm_deepseek.py`：DeepSeek 客户端
- `tts_melo.py`：MeloTTS（默认）
- `tts_cosyvoice.py`：CosyVoice2 TTS（固定音色）
- `voice_chat.py`：主入口：实时语音聊天

## 运行（最小化指引）

### 1) 环境变量

DeepSeek：

- `DEEPSEEK_API_KEY`：必填
- `DEEPSEEK_MODEL`：可选，默认 `deepseek-chat`
- `DEEPSEEK_BASE_URL`：可选，默认 `https://api.deepseek.com`

TTS 选择：

- `TTS_ENGINE=melo`（默认；更适合中文/中英混合，CPU 也能跑）
- `TTS_ENGINE=matcha`（Matcha-TTS 预训练英文模型，会自动下载 ckpt + vocoder）
- `TTS_ENGINE=cosyvoice`
- `TTS_ENGINE=openvoice`

CosyVoice（固定音色，必填）：

- `COSYVOICE_PROMPT_WAV`：一段 3~10 秒的参考音频（推荐 wav）
- `COSYVOICE_PROMPT_TEXT`：参考音频里说的文本（尽量一致）

### 2) 模型目录（不提交到 git）

把模型放到 `model/`（示例）：

- `model/silero_vad/silero_vad.onnx`
- `model/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/`（encoder/decoder/joiner/tokens 等）
- `model/vits-melo-tts-zh_en/`（model.onnx / lexicon / tokens / dict 等）
- `model/CosyVoice2-0.5B/`（CosyVoice2 模型目录）
- `model/openvoice_v2/checkpoints_v2/`（OpenVoiceV2 checkpoints：converter 等）

### 3) 启动语音聊天

推荐（带唤醒词，“旺财”唤醒；说“休眠/退出/再见”回到待机）：

```powershell
python .\main.py
```

可选：侧边悬浮 UI（点击展开/收起，支持托盘图标，可选麦克风/灵敏度/Provider 等，适合常驻）：

```powershell
python .\floating_ui.py
```

可选：普通窗口 UI（带日志区，便于调试；可选输入/输出设备）：

```powershell
python .\ui_window.py
```

托盘图标依赖（如未安装会自动提示）：

```powershell
pip install pystray
```

始终开启（不走唤醒词，调试用）：

```powershell
python .\voice_chat.py
```

CosyVoice：

```powershell
$env:TTS_ENGINE="cosyvoice"
$env:COSYVOICE_PROMPT_WAV="E:\path\to\ref.wav"
$env:COSYVOICE_PROMPT_TEXT="（ref.wav里说的内容）"
python .\main.py
```

OpenVoice V2：

推荐：OpenVoice 只负责“转音色”，底座 TTS 使用本项目的 MeloTTS。

依赖安装：

```powershell
$env:NO_PROXY="*"; $env:no_proxy="*"
# 本仓库已 vendored 了 OpenVoice 的 python 源码：third_party/openvoice_min
# 所以一般不需要额外安装 openvoice。
# 如果你想自己安装（不推荐拉全量依赖）：
#   pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
```

模型下载（把 huggingface 的 `myshell-ai/OpenVoiceV2` 下载到本地）：

```powershell
pip install huggingface-hub
hf download myshell-ai/OpenVoiceV2 --local-dir .\model\openvoice_v2\checkpoints_v2
```

底座 TTS（MeloTTS）模型下载（OpenVoice 需要它来先合成“源音色”再转音色）：

```powershell
python .\download_melo_model.py
```

启动：

```powershell
$env:TTS_ENGINE="openvoice"
$env:OPENVOICE_REF_WAV="E:\path\to\ref.wav"
# 推荐：底座用 melo（默认就是 melo）
$env:OPENVOICE_BASE_ENGINE="melo"
# 可选：$env:OPENVOICE_DEVICE="auto"  # 或 cuda / cpu
python .\main.py
```

MeloTTS（推荐：中文/中英混合，CPU 友好）：

先下载模型（约 167MB，会解压到 `model/vits-melo-tts-zh_en/`，已 gitignore）：

```powershell
python .\download_melo_model.py
```

启动：

```powershell
$env:TTS_ENGINE="melo"
python .\main.py
```

Matcha-TTS（英文预训练模型）：

注意：Matcha-TTS 这套预训练 checkpoint 是英文（LJSpeech / VCTK），中文输入不一定自然；如果你只是想要更像真人的中文 TTS，建议优先用 Melo / CosyVoice / OpenVoice。

```powershell
$env:TTS_ENGINE="matcha"
# Windows phonemizer 需要 espeak-ng；请自行安装 espeak-ng（或准备 espeak-ng.dll + espeak-ng-data）。
# 如果 phonemizer 找不到 espeak-ng，可手动指定：
# $env:PHONEMIZER_ESPEAK_LIBRARY="C:\\path\\to\\espeak-ng.dll"
# $env:ESPEAK_DATA_PATH="C:\\path\\to\\espeak-ng-data"
python .\main.py
```

## 注意事项

- `.gitignore` 已忽略 `model/`、音频文件、`output/`、`.venv/`、`.cache/`、`.env*` 等，避免提交大文件/敏感信息。
- 参考音频如果是 `m4a`，建议用 `ffmpeg` 转成 `wav`（24k/mono）再用作 `COSYVOICE_PROMPT_WAV`。
- 中文 TTS（特别是 MeloTTS）对“阿拉伯数字 / markdown / 生僻符号”可能不友好：本项目默认会做一层播报文本归一化（数字转中文、去 markdown）。可通过 `TTS_TEXT_NORMALIZE=0` 关闭。
- 这是原型工程，后续会继续模块化与整理依赖/文档。

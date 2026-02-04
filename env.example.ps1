$ErrorActionPreference = "Stop"

# Copy this file to env.ps1 and fill in your own keys, then run:
#   . .\\env.ps1
#
# IMPORTANT:
# - Do NOT commit env.ps1 (already ignored by .gitignore via *.env rules if you rename to .env,
#   or keep it untracked).
# - Keys are secrets.

# --- Secrets ---
# $env:PICOVOICE_ACCESS_KEY = "pv_xxx"
# $env:DEEPSEEK_API_KEY = "sk_xxx"

# --- Optional: proxy (if you are behind a VPN/proxy like Clash) ---
# Use HTTP proxy scheme here (common for 7890).
# $env:HTTP_PROXY  = "http://127.0.0.1:7890"
# $env:HTTPS_PROXY = "http://127.0.0.1:7890"
# $env:NO_PROXY    = "127.0.0.1,localhost"

# --- Engine selection ---
# piper (sherpa-onnx OfflineTts) | piper_native (official piper.exe) | melo | matcha | cosyvoice | openvoice
$env:TTS_ENGINE = "piper"

# --- Piper (official runtime) ---
# Only needed when $env:TTS_ENGINE = "piper_native"
# $env:PIPER_BIN = "E:\\path\\to\\piper.exe"
# $env:PIPER_NATIVE_MODEL_DIR = "E:\\Projects\\wangcai-assist\\model\\piper_zh_xiao_ya"   # voice folder containing *.onnx (+ *.onnx.json)
# Optional (when the folder contains multiple *.onnx):
# $env:PIPER_NATIVE_ONNX = "E:\\Projects\\wangcai-assist\\model\\piper_zh_xiao_ya\\zh_CN-xiao_ya.onnx"

# --- OpenVoice V2 ---
# If not set, tts_openvoice.py will use .\\myvoice.wav when it exists.
# $env:OPENVOICE_REF_WAV = "E:\\Projects\\wangcai-assist\\myvoice.wav"
# $env:OPENVOICE_DEVICE = "auto"          # auto | cpu | cuda
# $env:OPENVOICE_BASE_ENGINE = "piper"    # only piper supported in this repo
# $env:OPENVOICE_PIPER_PROVIDER = "cpu"   # cpu | cuda (if sherpa-onnx supports it)

# --- Matcha-TTS (pretrained English checkpoints) ---
# $env:TTS_ENGINE = "matcha"
# Models (ckpt + vocoder) will be downloaded to MATCHA_MODEL_DIR (default: .\\model\\matcha, gitignored).
# $env:MATCHA_MODEL_DIR = "E:\\Projects\\wangcai-assist\\model\\matcha"
# $env:MATCHA_MODEL = "matcha_ljspeech"     # matcha_ljspeech | matcha_vctk
# $env:MATCHA_DEVICE = "auto"              # auto | cuda | cpu
# $env:MATCHA_SPEAKER = "0"                # only for matcha_vctk (0~107)
# $env:MATCHA_STEPS = "10"
# $env:MATCHA_TEMPERATURE = "0.667"
# $env:MATCHA_SPEAKING_RATE = "0.95"       # higher => slower (Matcha length_scale)
# $env:MATCHA_DENOISER_STRENGTH = "0.00025"  # 0 disables denoiser (faster)
#
# Windows 需要 espeak-ng（phonemizer 后端）:
# 如果你已经下载了 Piper runtime（third_party\\piper\\piper\\espeak-ng.dll），本项目会自动复用。
# 也可以手动指定：
# $env:PHONEMIZER_ESPEAK_LIBRARY = "E:\\Projects\\wangcai-assist\\third_party\\piper\\piper\\espeak-ng.dll"
# $env:ESPEAK_DATA_PATH = "E:\\Projects\\wangcai-assist\\third_party\\piper\\piper\\espeak-ng-data"

# --- MeloTTS (sherpa-onnx VITS, zh_en) ---
# $env:TTS_ENGINE = "melo"
# $env:MELO_MODEL_DIR = "E:\\Projects\\wangcai-assist\\model\\vits-melo-tts-zh_en"
# 默认优先使用 model.int8.onnx（更省资源/更快）：
# $env:MELO_PREFER_INT8 = "1"   # 0 关闭
# 可选：指定具体 onnx
# $env:MELO_ONNX = "E:\\Projects\\wangcai-assist\\model\\vits-melo-tts-zh_en\\model.int8.onnx"

# --- Audio device selection ---
# -1 means default device.
# $env:MIC_DEVICE_INDEX = "-1"
# $env:KWS_DEVICE_INDEX = "-1"
# $env:KWS_SENSITIVITY = "0.5"
# Output speaker selection (only works when sounddevice is installed; otherwise uses system default):
# $env:OUTPUT_DEVICE_INDEX = "-1"

# --- ASR provider ---
# cpu | cuda (note: sherpa-onnx may fallback to cpu if GPU is not enabled in your build)
# $env:ASR_PROVIDER = "cuda"

# --- TTS text normalization (recommended for Chinese TTS) ---
# Some TTS models (e.g. MeloTTS) may not handle Arabic numerals / markdown well.
# When enabled, we will:
# - strip markdown markers
# - convert Arabic numerals to Chinese spoken forms (1 -> 一, 2026 -> 二零二六, 3.14 -> 三点一四, 50% -> 百分之五十)
# - optionally strip rare symbols/emojis
#
# $env:TTS_TEXT_NORMALIZE = "1"     # 1 enable (default), 0 disable
# $env:TTS_STRIP_NON_TEXT = "1"     # 1 enable (default), 0 keep all characters

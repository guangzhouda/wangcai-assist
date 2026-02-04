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
# melo (default) | cosyvoice | openvoice | matcha (English)
$env:TTS_ENGINE = "melo"

# --- OpenVoice V2 ---
# If not set, tts_openvoice.py will use .\\myvoice.wav when it exists.
# $env:OPENVOICE_REF_WAV = "E:\\Projects\\wangcai-assist\\myvoice.wav"
# $env:OPENVOICE_DEVICE = "auto"          # auto | cpu | cuda
# $env:OPENVOICE_BASE_ENGINE = "melo"     # base TTS used before voice conversion

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
# 你需要提供 espeak-ng 的 dll 与 data 目录（或自行安装到系统环境里）。
# 可手动指定：
# $env:PHONEMIZER_ESPEAK_LIBRARY = "C:\\path\\to\\espeak-ng.dll"
# $env:ESPEAK_DATA_PATH = "C:\\path\\to\\espeak-ng-data"

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

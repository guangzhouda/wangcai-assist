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
# piper | cosyvoice | openvoice
$env:TTS_ENGINE = "piper"

# --- OpenVoice V2 ---
# If not set, tts_openvoice.py will use .\\myvoice.wav when it exists.
# $env:OPENVOICE_REF_WAV = "E:\\Projects\\wangcai-assist\\myvoice.wav"
# $env:OPENVOICE_DEVICE = "auto"          # auto | cpu | cuda
# $env:OPENVOICE_BASE_ENGINE = "piper"    # only piper supported in this repo
# $env:OPENVOICE_PIPER_PROVIDER = "cpu"   # cpu | cuda (if sherpa-onnx supports it)

# --- Audio device selection ---
# -1 means default device.
# $env:MIC_DEVICE_INDEX = "-1"
# $env:KWS_DEVICE_INDEX = "-1"
# $env:KWS_SENSITIVITY = "0.5"

# --- ASR provider ---
# cpu | cuda (note: sherpa-onnx may fallback to cpu if GPU is not enabled in your build)
# $env:ASR_PROVIDER = "cuda"


"""Download Matcha-TTS pretrained checkpoints to a local folder (gitignored).

Usage (PowerShell):
  # optional: choose a local dir (default: .\\model\\matcha)
  $env:MATCHA_MODEL_DIR = "E:\\Projects\\wangcai-assist\\model\\matcha"
  $env:MATCHA_MODEL = "matcha_ljspeech"   # or matcha_vctk
  python .\\download_matcha_models.py

Notes:
  - These are English checkpoints (LJSpeech / VCTK).
  - The downloaded files are large and are ignored by git via .gitignore.
"""

import os
from pathlib import Path


MATCHA_URLS = {
    "matcha_ljspeech": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt",
    "matcha_vctk": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_vctk.ckpt",
}

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
}

MODEL_DEFAULTS = {
    "matcha_ljspeech": {"vocoder": "hifigan_T2_v1"},
    "matcha_vctk": {"vocoder": "hifigan_univ_v1"},
}


def main() -> None:
    model_dir = Path(os.environ.get("MATCHA_MODEL_DIR", "").strip() or (Path(__file__).resolve().parent / "model" / "matcha"))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("MATCHA_MODEL", "matcha_ljspeech").strip() or "matcha_ljspeech"
    if model_name not in MATCHA_URLS:
        raise SystemExit(f"Invalid MATCHA_MODEL={model_name!r}, expected one of: {', '.join(MATCHA_URLS.keys())}")

    vocoder_name = os.environ.get("MATCHA_VOCODER", "").strip() or MODEL_DEFAULTS[model_name]["vocoder"]
    if vocoder_name not in VOCODER_URLS:
        raise SystemExit(f"Invalid MATCHA_VOCODER={vocoder_name!r}, expected one of: {', '.join(VOCODER_URLS.keys())}")

    ckpt_path = model_dir / f"{model_name}.ckpt"
    voc_path = model_dir / vocoder_name

    try:
        from matcha.utils.utils import assert_model_downloaded  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing dependency: matcha-tts. Please run: pip install -U matcha-tts") from exc

    print(f"Downloading to: {model_dir}")
    assert_model_downloaded(ckpt_path, MATCHA_URLS[model_name])
    assert_model_downloaded(voc_path, VOCODER_URLS[vocoder_name])

    print("\nDone.")
    print(f"- Matcha ckpt: {ckpt_path}")
    print(f"- Vocoder:     {voc_path}")


if __name__ == "__main__":
    main()


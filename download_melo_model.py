"""Download sherpa-onnx VITS MeloTTS (zh_en) model to local `model/` (gitignored).

This model supports Chinese (and some English) and runs well on CPU.

Usage (PowerShell):
  python .\\download_melo_model.py

Optional env:
  - MELO_MODEL_DIR: target folder (default: .\\model\\vits-melo-tts-zh_en)
"""

import os
import tarfile
from pathlib import Path


URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"


def _download(url: str, dst: Path) -> None:
    import requests  # type: ignore

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    # Avoid broken local proxy envs (common on Windows with Clash/VPN).
    s = requests.Session()
    s.trust_env = False

    with s.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    model_dir = Path(os.environ.get("MELO_MODEL_DIR", "").strip() or (project_dir / "model" / "vits-melo-tts-zh_en"))
    model_dir = model_dir.expanduser()

    # Already there?
    if (model_dir / "tokens.txt").exists() and ((model_dir / "model.int8.onnx").exists() or (model_dir / "model.onnx").exists()):
        print(f"[+] Model already present: {model_dir}")
        return

    downloads = project_dir / "model" / "_tmp_sherpa"
    downloads.mkdir(parents=True, exist_ok=True)
    tar_path = downloads / "vits-melo-tts-zh_en.tar.bz2"

    print(f"[-] Downloading: {URL}")
    _download(URL, tar_path)
    print(f"[+] Downloaded: {tar_path} ({tar_path.stat().st_size} bytes)")

    # Extract. The archive contains a top-level folder "vits-melo-tts-zh_en/".
    extract_root = model_dir.parent
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"[-] Extracting to: {extract_root}")
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(path=extract_root)

    # Validate
    if not model_dir.exists():
        raise SystemExit(f"Extraction finished but target folder missing: {model_dir}")

    needed = [
        model_dir / "tokens.txt",
        model_dir / "lexicon.txt",
        model_dir / "model.onnx",
    ]
    if not all(p.exists() for p in needed):
        raise SystemExit(f"Model extracted but files look incomplete: {model_dir}")

    print("\nDone.")
    print(f"- Model dir: {model_dir}")
    print("Tip: 如果目录里存在有效的 model.int8.onnx，会优先使用；若是 Git-LFS 指针文件会自动回退到 model.onnx。")


if __name__ == "__main__":
    main()

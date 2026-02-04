import os
import wave
from pathlib import Path
from typing import Optional, Tuple

import sherpa_onnx


# sherpa-onnx VITS MeloTTS (zh_en) model (downloaded under model/, gitignored)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model" / "vits-melo-tts-zh_en"


def _safe_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _resolve_model_dir() -> Path:
    v = os.environ.get("MELO_MODEL_DIR", "").strip()
    if v:
        return Path(v).expanduser()
    return DEFAULT_MODEL_DIR


def _is_git_lfs_pointer(p: Path) -> bool:
    """Some release archives may contain Git-LFS pointer files instead of real weights."""
    try:
        if not p.exists() or not p.is_file():
            return False
        if p.stat().st_size > 1024 * 1024:  # real model files are large
            return False
        head = p.read_text(encoding="utf-8", errors="ignore")[:200]
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _resolve_paths(model_dir: Path) -> Tuple[Path, Path, Path, str, str]:
    model_dir = model_dir.expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"找不到 MeloTTS 模型目录：{model_dir}\n"
            "请先下载模型（参考 README / download_melo_model.py），或设置环境变量 MELO_MODEL_DIR。"
        )

    model_path_env = os.environ.get("MELO_ONNX", "").strip()
    if model_path_env:
        model_path = Path(model_path_env).expanduser()
        if not model_path.is_absolute():
            model_path = (model_dir / model_path).resolve()
    else:
        # Prefer int8 model on CPU
        prefer_int8 = os.environ.get("MELO_PREFER_INT8", "1").strip().lower() not in ("0", "false", "no")
        int8_path = model_dir / "model.int8.onnx"
        if prefer_int8 and int8_path.exists() and not _is_git_lfs_pointer(int8_path):
            model_path = int8_path
        elif (model_dir / "model.onnx").exists():
            model_path = model_dir / "model.onnx"
        else:
            # last resort: search any onnx in dir
            cand = sorted([p for p in model_dir.glob("*.onnx") if p.is_file()])
            if len(cand) != 1:
                raise FileNotFoundError(
                    f"Missing/ambiguous Melo *.onnx in: {model_dir} (found {len(cand)})\n"
                    "请设置环境变量 MELO_ONNX 指向具体的 .onnx 文件。"
                )
            model_path = cand[0]

    tokens = model_dir / "tokens.txt"
    lexicon = model_dir / "lexicon.txt"

    if not tokens.exists():
        raise FileNotFoundError(f"Missing tokens.txt: {tokens}")
    if not lexicon.exists():
        raise FileNotFoundError(f"Missing lexicon.txt: {lexicon}")

    # Melo model folder also contains FSTs + dict/ for text normalization.
    # sherpa-onnx uses dict_dir for these resources.
    dict_dir = str(model_dir)
    data_dir = ""  # not using espeak-ng-data for this model

    if not model_path.exists():
        raise FileNotFoundError(f"Missing Melo model: {model_path}")

    return model_path, tokens, lexicon, dict_dir, data_dir


def create_tts(
    *,
    provider: str = "cpu",
    num_threads: int = 2,
    noise_scale: Optional[float] = None,
    noise_scale_w: Optional[float] = None,
    length_scale: Optional[float] = None,
) -> sherpa_onnx.OfflineTts:
    model_dir = _resolve_model_dir()
    model_path, tokens_path, lexicon_path, dict_dir, data_dir = _resolve_paths(model_dir)

    # Allow env overrides for voice clarity / pace tuning.
    if noise_scale is None:
        noise_scale = _safe_float(os.environ.get("MELO_NOISE_SCALE", "").strip(), 0.667)
    if noise_scale_w is None:
        noise_scale_w = _safe_float(os.environ.get("MELO_NOISE_SCALE_W", "").strip(), 0.8)
    if length_scale is None:
        length_scale = _safe_float(os.environ.get("MELO_LENGTH_SCALE", "").strip(), 1.0)

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model_path),
        tokens=str(tokens_path),
        lexicon=str(lexicon_path),
        dict_dir=dict_dir,
        data_dir=data_dir,
        noise_scale=float(noise_scale),
        noise_scale_w=float(noise_scale_w),
        length_scale=float(length_scale),
    )

    model_cfg = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits,
        provider=provider,
        num_threads=num_threads,
        debug=False,
    )

    cfg = sherpa_onnx.OfflineTtsConfig(model=model_cfg)
    return sherpa_onnx.OfflineTts(cfg)


def synthesize_to_wav_with_duration(
    tts: sherpa_onnx.OfflineTts,
    text: str,
    wav_path: str,
    *,
    sid: int = 0,
    speed: float = 1.0,
) -> tuple[str, float]:
    audio = tts.generate(text, sid=sid, speed=speed)
    duration_sec = 0.0
    if audio.sample_rate > 0:
        duration_sec = len(audio.samples) / float(audio.sample_rate)

    ok = sherpa_onnx.write_wave(wav_path, audio.samples, audio.sample_rate)
    if not ok:
        raise RuntimeError("Failed to write wav file.")

    return wav_path, duration_sec

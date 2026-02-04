import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# NOTE:
# Matcha-TTS uses `phonemizer` (espeak backend) for english_cleaners2.
# On Windows, you must provide an espeak-ng shared library (dll) + data dir.
# If you already have Piper runtime downloaded, we can reuse its espeak-ng files.


SAMPLE_RATE = 22050


MATCHA_URLS = {
    "matcha_ljspeech": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt",
    "matcha_vctk": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_vctk.ckpt",
}

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
}

MODEL_DEFAULTS = {
    # Single speaker (English)
    "matcha_ljspeech": {"vocoder": "hifigan_T2_v1", "speaking_rate": 0.95, "spk": None},
    # Multi speaker (English)
    "matcha_vctk": {"vocoder": "hifigan_univ_v1", "speaking_rate": 0.85, "spk": 0},
}


def _safe_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: str, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _resolve_model_dir() -> Path:
    v = os.environ.get("MATCHA_MODEL_DIR", "").strip()
    if v:
        return Path(v).expanduser()
    # fallback to a project-local folder (gitignored)
    return Path(__file__).resolve().parent / "model" / "matcha"


def _ensure_espeak_env() -> None:
    """Make phonemizer(espeak) available on Windows.

    If user already configured env vars, respect them. Otherwise, try to reuse
    Piper runtime's bundled espeak-ng dll + data dir.
    """
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY", "").strip():
        return

    base = Path(__file__).resolve().parent
    dll = base / "third_party" / "piper" / "piper" / "espeak-ng.dll"
    data = base / "third_party" / "piper" / "piper" / "espeak-ng-data"
    if dll.exists() and data.exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(dll)
        os.environ.setdefault("ESPEAK_DATA_PATH", str(data))


def _write_wav_int16(wav_path: str, pcm16, sample_rate: int) -> None:
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())


@dataclass
class MatchaTts:
    device: str
    model_name: str
    ckpt_path: Path
    vocoder_name: str
    vocoder_path: Path
    speaker_id: Optional[int]
    steps: int
    temperature: float
    base_length_scale: float
    denoiser_strength: float

    _torch_device: object = None
    _model: object = None
    _vocoder: object = None
    _denoiser: object = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._vocoder is not None:
            return

        _ensure_espeak_env()

        import torch  # type: ignore

        from matcha.hifigan.config import v1  # type: ignore
        from matcha.hifigan.denoiser import Denoiser  # type: ignore
        from matcha.hifigan.env import AttrDict  # type: ignore
        from matcha.hifigan.models import Generator as HiFiGAN  # type: ignore
        from matcha.models.matcha_tts import MatchaTTS  # type: ignore

        if (self.device or "").strip().lower() in ("cuda", "gpu") and torch.cuda.is_available():
            self._torch_device = torch.device("cuda")
        elif (self.device or "").strip().lower() == "cpu":
            self._torch_device = torch.device("cpu")
        else:
            # auto
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = MatchaTTS.load_from_checkpoint(str(self.ckpt_path), map_location=self._torch_device)
        self._model.eval()

        # HiFiGAN vocoder
        h = AttrDict(v1)
        voc = HiFiGAN(h).to(self._torch_device)
        voc.load_state_dict(torch.load(str(self.vocoder_path), map_location=self._torch_device)["generator"])
        voc.eval()
        voc.remove_weight_norm()
        self._vocoder = voc

        if self.denoiser_strength and self.denoiser_strength > 0:
            self._denoiser = Denoiser(self._vocoder, mode="zeros")
        else:
            self._denoiser = None

    def generate(self, text: str, *, speed: float = 1.0) -> Tuple["object", int]:
        """Return (waveform_float32, sample_rate)."""
        self._ensure_loaded()

        text = (text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        import torch  # type: ignore
        import numpy as np  # type: ignore

        from matcha.text import text_to_sequence  # type: ignore
        from matcha.utils.utils import intersperse  # type: ignore

        length_scale = float(self.base_length_scale or 1.0)
        if speed and speed > 0:
            length_scale = length_scale / float(speed)

        x = torch.tensor(
            intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
            dtype=torch.long,
            device=self._torch_device,
        )[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=self._torch_device)
        spks = (
            torch.tensor([int(self.speaker_id)], dtype=torch.long, device=self._torch_device)
            if self.speaker_id is not None
            else None
        )

        with torch.inference_mode():
            out = self._model.synthesise(
                x,
                x_lengths,
                n_timesteps=int(self.steps),
                temperature=float(self.temperature),
                spks=spks,
                length_scale=float(length_scale),
            )
            mel = out["mel"]
            audio = self._vocoder(mel).clamp(-1, 1)
            if self._denoiser is not None:
                audio = self._denoiser(audio.squeeze(), strength=float(self.denoiser_strength)).cpu().squeeze()
            else:
                audio = audio.cpu().squeeze()

        wav = audio.detach().cpu().float().numpy()
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        return wav, SAMPLE_RATE

    def synthesize_to_wav(self, text: str, wav_path: str, *, speed: float = 1.0) -> Tuple[str, float]:
        import numpy as np  # type: ignore

        wav, sr = self.generate(text, speed=speed)
        pcm16 = (wav * 32767.0).astype(np.int16)
        _write_wav_int16(wav_path, pcm16, sr)
        dur = float(len(pcm16)) / float(sr) if sr > 0 else 0.0
        return wav_path, dur


def create_tts() -> MatchaTts:
    """Create a Matcha-TTS engine (pretrained English checkpoints).

    Env:
      - MATCHA_MODEL_DIR: folder for checkpoints (default: .\\model\\matcha)
      - MATCHA_MODEL: matcha_ljspeech | matcha_vctk (default: matcha_ljspeech)
      - MATCHA_DEVICE: auto | cuda | cpu
      - MATCHA_SPEAKER: speaker id (only for matcha_vctk)
      - MATCHA_STEPS: ODE steps (default: 10)
      - MATCHA_TEMPERATURE: default 0.667
      - MATCHA_SPEAKING_RATE: base length_scale (default depends on model)
      - MATCHA_DENOISER_STRENGTH: 0 disables denoiser (default 0.00025)
    """
    model_dir = _resolve_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("MATCHA_MODEL", "matcha_ljspeech").strip() or "matcha_ljspeech"
    if model_name not in MATCHA_URLS:
        raise ValueError(f"Invalid MATCHA_MODEL={model_name!r}, expected one of: {', '.join(MATCHA_URLS.keys())}")

    device = os.environ.get("MATCHA_DEVICE", "auto").strip() or "auto"

    defaults = MODEL_DEFAULTS.get(model_name, {})
    vocoder_name = os.environ.get("MATCHA_VOCODER", "").strip() or str(defaults.get("vocoder") or "hifigan_T2_v1")
    if vocoder_name not in VOCODER_URLS:
        raise ValueError(
            f"Invalid MATCHA_VOCODER={vocoder_name!r}, expected one of: {', '.join(VOCODER_URLS.keys())}"
        )

    speaker_id: Optional[int] = None
    v = os.environ.get("MATCHA_SPEAKER", "").strip()
    if v:
        speaker_id = _safe_int(v, 0)
    elif defaults.get("spk") is not None:
        speaker_id = int(defaults["spk"])

    steps = _safe_int(os.environ.get("MATCHA_STEPS", "").strip(), 10)
    temperature = _safe_float(os.environ.get("MATCHA_TEMPERATURE", "").strip(), 0.667)
    base_length_scale = _safe_float(
        os.environ.get("MATCHA_SPEAKING_RATE", "").strip(),
        float(defaults.get("speaking_rate") or 1.0),
    )
    denoiser_strength = _safe_float(os.environ.get("MATCHA_DENOISER_STRENGTH", "").strip(), 0.00025)

    ckpt_path = model_dir / f"{model_name}.ckpt"
    vocoder_path = model_dir / vocoder_name

    # Download if missing (to project-local model dir, gitignored).
    try:
        from matcha.utils.utils import assert_model_downloaded  # type: ignore
    except Exception as exc:
        raise RuntimeError("缺少 matcha-tts 依赖，无法下载/加载模型。请先 pip install -U matcha-tts") from exc

    assert_model_downloaded(ckpt_path, MATCHA_URLS[model_name])
    assert_model_downloaded(vocoder_path, VOCODER_URLS[vocoder_name])

    return MatchaTts(
        device=device,
        model_name=model_name,
        ckpt_path=ckpt_path,
        vocoder_name=vocoder_name,
        vocoder_path=vocoder_path,
        speaker_id=speaker_id,
        steps=steps,
        temperature=temperature,
        base_length_scale=base_length_scale,
        denoiser_strength=denoiser_strength,
    )


def synthesize_to_wav_with_duration(
    tts: MatchaTts,
    text: str,
    wav_path: str,
    *,
    speed: float = 1.0,
) -> tuple[str, float]:
    return tts.synthesize_to_wav(text, wav_path, speed=speed)


import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


DEFAULT_CKPT_DIR = Path(__file__).resolve().parent / "model" / "openvoice_v2" / "checkpoints_v2"


@dataclass
class GeneratedAudio:
    samples: np.ndarray
    sample_rate: int


def _read_wav_info(path: str) -> tuple[int, int]:
    info = sf.info(path)
    return int(info.samplerate), int(info.frames)


def _normalize_speaker_key(key: str) -> str:
    # "EN-US" -> "en-us", "EN_INDIA" -> "en-india", "ZH" -> "zh"
    return (key or "").strip().lower().replace("_", "-")


def _pick_default_speaker_key(language: str) -> str:
    lang = (language or "").strip().upper()
    if lang == "EN":
        return "EN-US"
    return lang or "ZH"


class OpenVoiceV2TTS:
    def __init__(
        self,
        *,
        ckpt_dir: str,
        ref_wav: str,
        language: str = "ZH",
        speaker_key: str = "",
        device: str = "auto",
        ref_vad: bool = True,
        tau: Optional[float] = None,
        watermark_message: str = "",
        speed: float = 1.0,
    ) -> None:
        # Lazy imports so "openvoice" stays optional.
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise SystemExit("缺少依赖 torch。请先安装 torch 后再使用 OpenVoice。") from exc

        try:
            from openvoice import se_extractor  # type: ignore
            from openvoice.api import ToneColorConverter  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "缺少依赖 openvoice。请先安装：pip install git+https://github.com/myshell-ai/OpenVoice.git"
            ) from exc

        try:
            from melo.api import TTS as MeloTTS  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "缺少依赖 MeloTTS。请先安装：pip install git+https://github.com/myshell-ai/MeloTTS.git"
            ) from exc

        self._torch = torch
        self._se_extractor = se_extractor
        self._ToneColorConverter = ToneColorConverter
        self._MeloTTS = MeloTTS

        self.ckpt_dir = str(Path(ckpt_dir).expanduser())
        self.ref_wav = str(Path(ref_wav).expanduser())
        self.language = (language or "ZH").strip().upper()
        self.speaker_key = (speaker_key or "").strip().upper()
        self.device = (device or "auto").strip().lower()
        self.ref_vad = bool(ref_vad)
        self.tau = tau
        self.watermark_message = (watermark_message or "").strip()
        self.speed = float(speed)

        ckpt = Path(self.ckpt_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing OpenVoice checkpoint dir: {ckpt}")

        converter_cfg = ckpt / "converter" / "config.json"
        converter_ckpt = ckpt / "converter" / "checkpoint.pth"
        if not converter_cfg.exists() or not converter_ckpt.exists():
            raise FileNotFoundError(
                "OpenVoice V2 converter files missing. Expect:\n"
                f"- {converter_cfg}\n"
                f"- {converter_ckpt}"
            )

        ref_path = Path(self.ref_wav)
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing OPENVOICE_REF_WAV: {ref_path}")

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        # Init converter
        self.converter = ToneColorConverter(str(converter_cfg), device=self._device)
        self.converter.load_ckpt(str(converter_ckpt))

        # Init base TTS (MeloTTS)
        self.melo = MeloTTS(language=self.language, device=self._device)
        self.spk2id = dict(self.melo.hps.data.spk2id)

        if not self.speaker_key:
            self.speaker_key = _pick_default_speaker_key(self.language)
        if self.speaker_key not in self.spk2id:
            raise ValueError(
                f"Invalid OPENVOICE_SPEAKER_KEY '{self.speaker_key}'. "
                f"Available: {sorted(self.spk2id.keys())}"
            )

        self._speaker_id = int(self.spk2id[self.speaker_key])

        # Load source speaker embedding (precomputed in OpenVoiceV2 checkpoints).
        se_key = _normalize_speaker_key(self.speaker_key)
        se_path = ckpt / "base_speakers" / "ses" / f"{se_key}.pth"
        if not se_path.exists():
            # Fallback for languages that only have one embedding in ckpt.
            se_path = ckpt / "base_speakers" / "ses" / f"{_normalize_speaker_key(self.language)}.pth"
        if not se_path.exists():
            raise FileNotFoundError(f"Missing base speaker embedding: {se_path}")
        self.src_se = torch.load(str(se_path), map_location=self._device).to(self._device)

        # Extract target speaker embedding from reference audio.
        # This is the voice you want to clone.
        processed_dir = Path(__file__).resolve().parent / "output" / "openvoice_processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        tgt = se_extractor.get_se(
            str(ref_path),
            self.converter,
            target_dir=str(processed_dir),
            vad=self.ref_vad,
        )
        if isinstance(tgt, (tuple, list)) and tgt:
            tgt_se = tgt[0]
        else:
            tgt_se = tgt
        self.tgt_se = tgt_se.to(self._device)

        # OpenVoice V2 commonly outputs 24kHz. We still read the actual sample rate
        # from the written wav when computing duration.
        self.sample_rate = 24000

    def _convert_to_file(self, *, audio_src_path: str, output_path: str) -> None:
        # API differs slightly across OpenVoice versions. Try with optional args first.
        kwargs = {
            "audio_src_path": audio_src_path,
            "src_se": self.src_se,
            "tgt_se": self.tgt_se,
            "output_path": output_path,
        }
        if self.tau is not None:
            kwargs["tau"] = float(self.tau)
        if self.watermark_message:
            kwargs["message"] = self.watermark_message

        try:
            self.converter.convert(**kwargs)
            return
        except TypeError:
            pass

        # Fallback: drop optional args.
        kwargs.pop("tau", None)
        try:
            self.converter.convert(**kwargs)
            return
        except TypeError:
            pass

        kwargs.pop("message", None)
        self.converter.convert(**kwargs)

    def synthesize_to_wav(self, text: str, wav_path: str, *, speed: float = 1.0) -> str:
        text = (text or "").strip()
        if not text:
            # Create an empty wav to keep callers simple.
            sf.write(wav_path, np.zeros((0,), dtype=np.float32), self.sample_rate)
            return wav_path

        out_dir = Path(wav_path).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        fd, base_wav = tempfile.mkstemp(prefix="ov2_base_", suffix=".wav", dir=str(out_dir))
        os.close(fd)
        try:
            # Base speech from MeloTTS (fast and stable). Output is then converted to target voice.
            self.melo.tts_to_file(
                text,
                self._speaker_id,
                base_wav,
                speed=float(speed),
            )
            self._convert_to_file(audio_src_path=base_wav, output_path=wav_path)
            return wav_path
        finally:
            try:
                os.remove(base_wav)
            except Exception:
                pass

    def generate(self, text: str) -> GeneratedAudio:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fd, wav_path = tempfile.mkstemp(prefix="ov2_", suffix=".wav", dir=str(out_dir))
        os.close(fd)
        try:
            self.synthesize_to_wav(text, wav_path, speed=self.speed)
            samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples[:, 0]
            return GeneratedAudio(samples=np.asarray(samples, dtype=np.float32), sample_rate=int(sr))
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass


def create_tts(
    *,
    ckpt_dir: Optional[str] = None,
    ref_wav: Optional[str] = None,
    language: Optional[str] = None,
    speaker_key: Optional[str] = None,
    device: Optional[str] = None,
    ref_vad: Optional[bool] = None,
    tau: Optional[float] = None,
    watermark_message: Optional[str] = None,
    speed: float = 1.0,
) -> OpenVoiceV2TTS:
    if ckpt_dir is None:
        ckpt_dir = os.environ.get("OPENVOICE_CKPT_DIR", str(DEFAULT_CKPT_DIR))
    if ref_wav is None:
        ref_wav = os.environ.get("OPENVOICE_REF_WAV", "")
    if not ref_wav:
        raise SystemExit("请先设置环境变量 OPENVOICE_REF_WAV (参考音频，用于固定音色)")

    if language is None:
        language = os.environ.get("OPENVOICE_LANGUAGE", "ZH")
    if speaker_key is None:
        speaker_key = os.environ.get("OPENVOICE_SPEAKER_KEY", "")
    if device is None:
        device = os.environ.get("OPENVOICE_DEVICE", "auto")
    if ref_vad is None:
        ref_vad = os.environ.get("OPENVOICE_REF_VAD", "1").strip() not in ("0", "false", "no")
    if tau is None:
        v = os.environ.get("OPENVOICE_TAU", "").strip()
        if v:
            try:
                tau = float(v)
            except ValueError:
                tau = None
    if watermark_message is None:
        watermark_message = os.environ.get("OPENVOICE_WATERMARK", "").strip()

    return OpenVoiceV2TTS(
        ckpt_dir=str(ckpt_dir),
        ref_wav=str(ref_wav),
        language=str(language),
        speaker_key=str(speaker_key or ""),
        device=str(device),
        ref_vad=bool(ref_vad),
        tau=tau,
        watermark_message=str(watermark_message or ""),
        speed=float(speed),
    )


def synthesize_to_wav_with_duration(
    tts: OpenVoiceV2TTS,
    text: str,
    wav_path: str,
    *,
    sid: int = 0,
    speed: float = 1.0,
) -> tuple[str, float]:
    # Keep signature compatible with other TTS modules.
    _ = sid
    wav_path = tts.synthesize_to_wav(text, wav_path, speed=speed)
    sr, frames = _read_wav_info(wav_path)
    dur = 0.0 if sr <= 0 else (frames / float(sr))
    return wav_path, dur

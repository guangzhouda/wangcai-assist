import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


# OpenVoice V2 checkpoints (HuggingFace: myshell-ai/OpenVoiceV2)
DEFAULT_CKPT_DIR = Path(__file__).resolve().parent / "model" / "openvoice_v2" / "checkpoints_v2"


@dataclass
class GeneratedAudio:
    samples: np.ndarray
    sample_rate: int


def _read_wav_info(path: str) -> Tuple[int, int]:
    info = sf.info(path)
    return int(info.samplerate), int(info.frames)


def _import_openvoice_core():
    """Import only the minimal OpenVoice python modules we need.

    IMPORTANT:
    - Do NOT import openvoice.api / openvoice.se_extractor here.
      They pull a lot of extra dependencies (faster-whisper/av, etc).
    """
    # Prefer the vendored OpenVoice python sources so users don't need to pip install
    # `git+https://github.com/myshell-ai/OpenVoice.git` (which often drags in heavy deps).
    vendored = Path(__file__).resolve().parent / "third_party" / "openvoice_min"
    if vendored.exists():
        vendored_str = str(vendored)
        if vendored_str not in sys.path:
            sys.path.insert(0, vendored_str)

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise SystemExit("缺少依赖 torch。请先安装 torch 后再使用 OpenVoice。") from exc

    try:
        import librosa  # type: ignore
    except Exception as exc:
        raise SystemExit("缺少依赖 librosa。请先安装：pip install librosa") from exc

    try:
        from openvoice import commons, utils  # type: ignore
        from openvoice.mel_processing import spectrogram_torch  # type: ignore
        from openvoice.models import SynthesizerTrn  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "缺少依赖 openvoice（仅需要其 python 代码，不需要安装一堆 demo 依赖）。\n"
            "如果你是从本仓库运行，优先使用 vendored 版本：third_party/openvoice_min\n"
            "如果你想自己安装：\n"
            "  python -m pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git"
        ) from exc

    return torch, librosa, utils, commons, spectrogram_torch, SynthesizerTrn


class _OpenVoiceBase:
    def __init__(self, config_path: str, *, device: str):
        torch, _, utils, _, _, SynthesizerTrn = _import_openvoice_core()

        if "cuda" in device:
            assert torch.cuda.is_available(), "CUDA 不可用，但 device 选择了 cuda"

        hps = utils.get_hparams_from_file(config_path)
        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
        model.eval()

        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path: str) -> None:
        torch, _, _, _, _, _ = _import_openvoice_core()
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint_dict["model"], strict=False)


class _ToneColorConverter(_OpenVoiceBase):
    def __init__(self, config_path: str, *, device: str, enable_watermark: bool = False):
        super().__init__(config_path, device=device)

        self.watermark_model = None
        if enable_watermark:
            try:
                import wavmark  # type: ignore

                self.watermark_model = wavmark.load_model().to(self.device)
            except Exception:
                # watermark is optional; if it can't be loaded, just disable.
                self.watermark_model = None

        self.version = getattr(self.hps, "_version_", "v2")

    def extract_se(self, ref_wav_list, se_save_path: Optional[str] = None):
        torch, librosa, _, _, spectrogram_torch, _ = _import_openvoice_core()

        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        device = self.device
        hps = self.hps
        gs = []

        for fname in ref_wav_list:
            audio_ref, _sr = librosa.load(fname, sr=hps.data.sampling_rate, mono=True)
            y = torch.FloatTensor(audio_ref).to(device).unsqueeze(0)
            y = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())

        gs = torch.stack(gs).mean(0)
        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)
        return gs

    def convert(
        self,
        *,
        audio_src_path: str,
        src_se,
        tgt_se,
        output_path: str,
        tau: float = 0.3,
        message: str = "default",
    ) -> None:
        torch, librosa, utils, _, spectrogram_torch, _ = _import_openvoice_core()

        hps = self.hps
        audio, _sr = librosa.load(audio_src_path, sr=hps.data.sampling_rate, mono=True)
        audio = torch.tensor(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device).unsqueeze(0)
            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            out = self.model.voice_conversion(
                spec,
                spec_lengths,
                sid_src=src_se,
                sid_tgt=tgt_se,
                tau=float(tau),
            )[0][0, 0].data.cpu().float().numpy()

            out = self._add_watermark(out, message, utils)
            sf.write(output_path, out, hps.data.sampling_rate)

    def _add_watermark(self, audio: np.ndarray, message: str, utils_mod) -> np.ndarray:
        if self.watermark_model is None:
            return audio
        torch, _, _, _, _, _ = _import_openvoice_core()

        bits = utils_mod.string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K : (coeff * n + 1) * K]
            if len(trunck) != K:
                break
            message_npy = bits[n * 32 : (n + 1) * 32]
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(self.device)[None]
                signal_wmd_tensor = self.watermark_model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K : (coeff * n + 1) * K] = signal_wmd_npy
        return audio


class OpenVoiceV2TTS:
    def __init__(
        self,
        *,
        ckpt_dir: str,
        ref_wav: str,
        device: str = "auto",
        speed: float = 1.0,
        tau: Optional[float] = None,
        watermark_message: str = "",
        base_engine: str = "piper",
        piper_provider: str = "cpu",
    ) -> None:
        torch, _, _, _, _, _ = _import_openvoice_core()

        self.ckpt_dir = str(Path(ckpt_dir).expanduser())
        self.ref_wav = str(Path(ref_wav).expanduser())
        self.device = (device or "auto").strip().lower()
        self.speed = float(speed)
        self.tau = tau
        self.watermark_message = (watermark_message or "").strip()
        self.base_engine = (base_engine or "piper").strip().lower()
        self.piper_provider = (piper_provider or "cpu").strip().lower()

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

        enable_watermark = bool(self.watermark_message)
        self.converter = _ToneColorConverter(
            str(converter_cfg),
            device=self._device,
            enable_watermark=enable_watermark,
        )
        self.converter.load_ckpt(str(converter_ckpt))

        # Embeddings:
        out_dir = Path(__file__).resolve().parent / "output" / "openvoice_processed"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.tgt_se = self.converter.extract_se(str(ref_path), se_save_path=str(out_dir / "tgt_se.pth")).to(self._device)

        # Base engine (default: Piper) + precompute src_se once
        if self.base_engine != "piper":
            raise SystemExit("当前仅支持 OPENVOICE_BASE_ENGINE=piper（更稳，不会引入额外依赖）。")

        from tts_piper import create_tts as piper_create_tts, synthesize_to_wav as piper_synthesize_to_wav

        self.piper_tts = piper_create_tts(provider=self.piper_provider)

        fd, probe_wav = tempfile.mkstemp(prefix="ov_src_", suffix=".wav", dir=str(out_dir))
        os.close(fd)
        try:
            piper_synthesize_to_wav(self.piper_tts, "你好", probe_wav, speed=1.0)
            self.src_se = self.converter.extract_se(probe_wav, se_save_path=str(out_dir / "src_se.pth")).to(self._device)
        finally:
            try:
                os.remove(probe_wav)
            except Exception:
                pass

        # OpenVoice V2 commonly uses 24kHz, but we always read actual SR from the written wav.
        self.sample_rate = int(getattr(self.converter.hps.data, "sampling_rate", 24000))

    def synthesize_to_wav(self, text: str, wav_path: str, *, speed: float = 1.0) -> str:
        text = (text or "").strip()
        if not text:
            sf.write(wav_path, np.zeros((0,), dtype=np.float32), self.sample_rate)
            return wav_path

        out_dir = Path(wav_path).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        from tts_piper import synthesize_to_wav as piper_synthesize_to_wav

        fd, base_wav = tempfile.mkstemp(prefix="ov_base_", suffix=".wav", dir=str(out_dir))
        os.close(fd)
        try:
            piper_synthesize_to_wav(self.piper_tts, text, base_wav, speed=float(speed))

            tau = float(self.tau) if self.tau is not None else 0.3
            self.converter.convert(
                audio_src_path=base_wav,
                src_se=self.src_se,
                tgt_se=self.tgt_se,
                output_path=wav_path,
                tau=tau,
                message=self.watermark_message or "default",
            )
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
    device: Optional[str] = None,
    tau: Optional[float] = None,
    watermark_message: Optional[str] = None,
    speed: float = 1.0,
    base_engine: Optional[str] = None,
    piper_provider: Optional[str] = None,
) -> OpenVoiceV2TTS:
    if ckpt_dir is None:
        ckpt_dir = os.environ.get("OPENVOICE_CKPT_DIR", str(DEFAULT_CKPT_DIR))
    if ref_wav is None:
        ref_wav = os.environ.get("OPENVOICE_REF_WAV", "")

    # Convenience: if user didn't set env var, but a local reference wav exists,
    # use it automatically to reduce setup friction on Windows.
    if not (ref_wav or "").strip():
        default_ref = Path(__file__).resolve().parent / "myvoice.wav"
        if default_ref.exists():
            ref_wav = str(default_ref)

    if not (ref_wav or "").strip():
        raise SystemExit("请先设置环境变量 OPENVOICE_REF_WAV (参考音频，用于固定音色)")

    if device is None:
        device = os.environ.get("OPENVOICE_DEVICE", "auto")
    if base_engine is None:
        base_engine = os.environ.get("OPENVOICE_BASE_ENGINE", "piper")
    if piper_provider is None:
        piper_provider = os.environ.get("OPENVOICE_PIPER_PROVIDER", "cpu")
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
        device=str(device),
        speed=float(speed),
        tau=tau,
        watermark_message=str(watermark_message or ""),
        base_engine=str(base_engine),
        piper_provider=str(piper_provider),
    )


def synthesize_to_wav_with_duration(
    tts: OpenVoiceV2TTS,
    text: str,
    wav_path: str,
    *,
    sid: int = 0,
    speed: float = 1.0,
) -> tuple[str, float]:
    _ = sid
    wav_path = tts.synthesize_to_wav(text, wav_path, speed=speed)
    sr, frames = _read_wav_info(wav_path)
    dur = 0.0 if sr <= 0 else (frames / float(sr))
    return wav_path, dur

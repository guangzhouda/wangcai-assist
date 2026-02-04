import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


COSYVOICE_ROOT = (
    Path(__file__).resolve().parent / "third_party" / "CosyVoice-src" / "CosyVoice-main"
)
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model" / "CosyVoice2-0.5B"


def _ensure_cosyvoice_on_path() -> None:
    # CosyVoice repo is vendored under third_party. Matcha-TTS is vendored as a
    # submodule substitute and is imported as "matcha".
    # Also keep HuggingFace caches inside the project by default, to avoid
    # permission issues writing to user profile directories on Windows.
    cache_root = Path(__file__).resolve().parent / ".cache" / "huggingface"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(cache_root)
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hub")
    try:
        Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    cosy = str(COSYVOICE_ROOT)
    matcha = str(MATCHA_ROOT)
    if matcha not in sys.path:
        sys.path.insert(0, matcha)
    if cosy not in sys.path:
        sys.path.insert(0, cosy)


@dataclass
class GeneratedAudio:
    samples: np.ndarray
    sample_rate: int


class CosyVoiceTTS:
    def __init__(
        self,
        *,
        model_dir: str,
        prompt_wav: str,
        prompt_text: str,
        spk_id: str = "default",
        fp16: bool = True,
        stream: bool = False,
        speed: float = 1.0,
        text_frontend: bool = True,
    ) -> None:
        _ensure_cosyvoice_on_path()
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore

        self.model_dir = model_dir
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.spk_id = spk_id
        self.stream = stream
        self.speed = speed
        self.text_frontend = text_frontend

        # Cache speaker features once; otherwise every chunk would re-extract
        # prompt tokens/embeddings from the reference audio (slow).
        self._cosy = CosyVoice2(model_dir, fp16=fp16)
        self.sample_rate = int(self._cosy.sample_rate)
        self._cosy.add_zero_shot_spk(prompt_text, prompt_wav, spk_id)

    def generate(self, text: str) -> GeneratedAudio:
        chunks: list[np.ndarray] = []
        for out in self._cosy.inference_zero_shot(
            text,
            # We already cached the speaker prompt as `spk_id` via add_zero_shot_spk().
            # Passing an empty prompt_text avoids "tts too short than prompt" warnings
            # when we intentionally chunk output for low latency.
            "",
            self.prompt_wav,
            zero_shot_spk_id=self.spk_id,
            stream=self.stream,
            speed=self.speed,
            text_frontend=self.text_frontend,
        ):
            t = out["tts_speech"]
            if hasattr(t, "detach"):
                arr = t.detach().cpu().squeeze(0).numpy()
            else:
                arr = np.asarray(t).squeeze()
            chunks.append(arr.astype(np.float32, copy=False))

        if chunks:
            samples = np.concatenate(chunks, axis=0)
        else:
            samples = np.zeros((0,), dtype=np.float32)

        return GeneratedAudio(samples=samples, sample_rate=self.sample_rate)


def _float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def create_tts(
    *,
    model_dir: Optional[str] = None,
    prompt_wav: Optional[str] = None,
    prompt_text: Optional[str] = None,
    spk_id: str = "default",
    fp16: bool = True,
    stream: bool = False,
    speed: float = 1.0,
    text_frontend: bool = True,
) -> CosyVoiceTTS:
    if model_dir is None:
        model_dir = os.environ.get("COSYVOICE_MODEL_DIR", str(DEFAULT_MODEL_DIR))

    if prompt_wav is None:
        prompt_wav = os.environ.get("COSYVOICE_PROMPT_WAV", "")

    if prompt_text is None:
        prompt_text = os.environ.get("COSYVOICE_PROMPT_TEXT", "你好，我是旺财。")

    if not prompt_wav:
        raise SystemExit(
            "请先设置环境变量 COSYVOICE_PROMPT_WAV (一段 3~10 秒的 wav 参考音频，用于固定音色)"
        )

    prompt_path = Path(prompt_wav).expanduser()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing COSYVOICE_PROMPT_WAV: {prompt_path}")

    model_path = Path(model_dir).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CosyVoice model dir: {model_path}")

    return CosyVoiceTTS(
        model_dir=str(model_path),
        prompt_wav=str(prompt_path),
        prompt_text=prompt_text,
        spk_id=spk_id,
        fp16=fp16,
        stream=stream,
        speed=speed,
        text_frontend=text_frontend,
    )


def synthesize_to_wav(
    tts: CosyVoiceTTS,
    text: str,
    wav_path: str,
) -> str:
    wav_path, _ = synthesize_to_wav_with_duration(tts, text, wav_path)
    return wav_path


def synthesize_to_wav_with_duration(
    tts: CosyVoiceTTS,
    text: str,
    wav_path: str,
    *,
    sid: int = 0,
    speed: float = 1.0,
) -> tuple[str, float]:
    # Keep signature compatible with other TTS backends used by voice_chat.py.
    # CosyVoice doesn't use "sid"; speed is supported via CosyVoiceTTS.speed.
    _ = sid

    old_speed = tts.speed
    if speed != old_speed:
        tts.speed = speed
    try:
        audio = tts.generate(text)
    finally:
        if tts.speed != old_speed:
            tts.speed = old_speed

    duration_sec = 0.0
    if audio.sample_rate > 0:
        duration_sec = len(audio.samples) / float(audio.sample_rate)

    samples_i16 = _float_to_int16(audio.samples)
    sf.write(wav_path, samples_i16, audio.sample_rate, subtype="PCM_16")
    return wav_path, duration_sec


def speak(
    text: str,
    *,
    tts: Optional[CosyVoiceTTS] = None,
    wav_path: Optional[str] = None,
    play: bool = True,
) -> str:
    if tts is None:
        tts = create_tts()

    if wav_path is None:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="tts_cosy_", suffix=".wav", dir=str(out_dir))
        os.close(fd)
        wav_path = tmp

    wav_path = synthesize_to_wav(tts, text, wav_path)

    if play:
        try:
            import winsound

            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        except Exception:
            pass

    return wav_path


def speak_with_typewriter(
    text: str,
    *,
    tts: Optional[CosyVoiceTTS] = None,
    prefix: str = "助手: ",
    end: str = "\n",
) -> str:
    """Play audio while printing text at (roughly) the same pace as the audio."""
    if tts is None:
        tts = create_tts()

    audio = tts.generate(text)
    duration_sec = 0.0
    if audio.sample_rate > 0:
        duration_sec = len(audio.samples) / float(audio.sample_rate)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, wav_path = tempfile.mkstemp(prefix="tts_cosy_", suffix=".wav", dir=str(out_dir))
    os.close(fd)

    samples_i16 = _float_to_int16(audio.samples)
    sf.write(wav_path, samples_i16, audio.sample_rate, subtype="PCM_16")

    try:
        import winsound

        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass

    sys.stdout.write(prefix)
    sys.stdout.flush()

    if not text or duration_sec <= 0:
        sys.stdout.write(text + end)
        sys.stdout.flush()
        return wav_path

    start = time.perf_counter()
    n = len(text)
    for i, ch in enumerate(text, start=1):
        sys.stdout.write(ch)
        sys.stdout.flush()

        target = start + duration_sec * (i / n)
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)

    sys.stdout.write(end)
    sys.stdout.flush()

    remaining = start + duration_sec - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)

    return wav_path


if __name__ == "__main__":
    # Quick manual test:
    #   set COSYVOICE_PROMPT_WAV=path\\to\\ref.wav
    #   python tts_cosyvoice.py
    tts = create_tts()
    speak_with_typewriter("你好，我是旺财。", tts=tts)

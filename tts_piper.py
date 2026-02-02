import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import sherpa_onnx


MODEL_DIR = Path(__file__).resolve().parent / "model" / "vits-piper-zh_CN-huayan-medium"
MODEL_PATH = MODEL_DIR / "zh_CN-huayan-medium.onnx"
TOKENS_PATH = MODEL_DIR / "tokens.txt"
DATA_DIR = MODEL_DIR / "espeak-ng-data"


def create_tts(
    *,
    provider: str = "cpu",
    num_threads: int = 2,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
) -> sherpa_onnx.OfflineTts:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing TTS model: {MODEL_PATH}")
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(f"Missing tokens.txt: {TOKENS_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing espeak-ng-data: {DATA_DIR}")

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(MODEL_PATH),
        tokens=str(TOKENS_PATH),
        data_dir=str(DATA_DIR),
        lexicon="",
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
        length_scale=length_scale,
    )

    model_cfg = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits,
        provider=provider,
        num_threads=num_threads,
        debug=False,
    )

    cfg = sherpa_onnx.OfflineTtsConfig(model=model_cfg)
    return sherpa_onnx.OfflineTts(cfg)


def synthesize_to_wav(
    tts: sherpa_onnx.OfflineTts,
    text: str,
    wav_path: str,
    *,
    sid: int = 0,
    speed: float = 1.0,
) -> str:
    wav_path, _ = synthesize_to_wav_with_duration(tts, text, wav_path, sid=sid, speed=speed)
    return wav_path


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


def speak(
    text: str,
    *,
    tts: Optional[sherpa_onnx.OfflineTts] = None,
    wav_path: Optional[str] = None,
    sid: int = 0,
    speed: float = 1.0,
    play: bool = True,
) -> str:
    """Synthesize text to a wav file and optionally play it (Windows only)."""
    if tts is None:
        tts = create_tts()

    if wav_path is None:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=str(out_dir))
        os.close(fd)
        wav_path = tmp

    wav_path = synthesize_to_wav(tts, text, wav_path, sid=sid, speed=speed)

    if play:
        try:
            import winsound

            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        except Exception:
            # If playback fails, we still keep the wav file for manual checking.
            pass

    return wav_path


def speak_with_typewriter(
    text: str,
    *,
    tts: Optional[sherpa_onnx.OfflineTts] = None,
    sid: int = 0,
    speed: float = 1.0,
    prefix: str = "助手: ",
    end: str = "\n",
) -> str:
    """Play audio while printing text at (roughly) the same pace as the audio.

    This is not true streaming TTS, but it solves the "text finishes earlier than audio"
    problem by throttling the console output to match the generated wav duration.
    """
    if tts is None:
        tts = create_tts()

    audio = tts.generate(text, sid=sid, speed=speed)
    duration_sec = 0.0
    if audio.sample_rate > 0:
        duration_sec = len(audio.samples) / float(audio.sample_rate)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, wav_path = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=str(out_dir))
    os.close(fd)

    ok = sherpa_onnx.write_wave(wav_path, audio.samples, audio.sample_rate)
    if not ok:
        raise RuntimeError("Failed to write wav file.")

    # Start playing in background so we can print while it is playing.
    try:
        import winsound

        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        # If playback fails, we still do typewriter output.
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

    # Ensure the audio has time to finish before we return.
    remaining = start + duration_sec - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)

    return wav_path

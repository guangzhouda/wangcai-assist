import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import sherpa_onnx


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model" / "vits-piper-zh_CN-huayan-medium"


def _resolve_model_dir() -> Path:
    v = os.environ.get("PIPER_MODEL_DIR", "").strip()
    if v:
        return Path(v).expanduser()
    return DEFAULT_MODEL_DIR


def _resolve_paths(model_dir: Path) -> Tuple[Path, Path, Path]:
    """Resolve (model.onnx, tokens.txt, espeak data_dir).

    We keep this flexible so you can download any Piper voice (onnx + onnx.json)
    into a folder and point PIPER_MODEL_DIR at it.
    """
    model_dir = model_dir.expanduser()
    model_path_env = os.environ.get("PIPER_ONNX", "").strip()
    if model_path_env:
        model_path = Path(model_path_env).expanduser()
        if not model_path.is_absolute():
            model_path = (model_dir / model_path).resolve()
    else:
        onnx_files = sorted([p for p in model_dir.glob("*.onnx") if p.is_file()])
        if not onnx_files:
            # If files were downloaded via `hf download`, they might live under a
            # nested subfolder that mirrors the repo structure.
            onnx_files = sorted(
                [
                    p
                    for p in model_dir.rglob("*.onnx")
                    if p.is_file() and ".cache" not in p.parts and ".git" not in p.parts
                ]
            )
        if len(onnx_files) != 1:
            raise FileNotFoundError(
                f"Missing/ambiguous *.onnx in: {model_dir} (found {len(onnx_files)})\n"
                "请设置环境变量 PIPER_ONNX 指向具体的 .onnx 文件。"
            )
        model_path = onnx_files[0]

    tokens_path = model_dir / "tokens.txt"
    if not tokens_path.exists():
        # Auto-generate tokens.txt from Piper's *.onnx.json if present.
        json_path = model_path.with_suffix(".onnx.json")
        if json_path.exists():
            try:
                import json

                obj = json.loads(json_path.read_text(encoding="utf-8"))

                # sherpa-onnx's built-in piper phonemizer expects eSpeak-style phonemes.
                # Some Piper voices are trained with `phoneme_type=pinyin` and are not
                # compatible with this pipeline.
                phoneme_type = (obj.get("phoneme_type") or "").strip().lower()
                if phoneme_type == "pinyin":
                    raise RuntimeError(
                        "该 Piper 音色的 phoneme_type=pinyin，当前 sherpa-onnx 的 piper phonemize 不支持，"
                        "会导致发音/加载失败。建议使用 zh_CN-huayan-*（espeak）或改用官方 piper 运行时。"
                    )

                phoneme_id_map = obj.get("phoneme_id_map") or {}
                if isinstance(phoneme_id_map, dict) and phoneme_id_map:
                    def _as_int(v) -> int:
                        if isinstance(v, list) and v:
                            v = v[0]
                        return int(v)

                    inv = sorted(((_as_int(i), str(t)) for t, i in phoneme_id_map.items()), key=lambda x: x[0])
                    tokens_path.write_text(
                        "".join(f"{tok} {idx}\n" for idx, tok in inv),
                        encoding="utf-8",
                    )
            except Exception:
                pass

    data_dir = model_dir / "espeak-ng-data"
    if not data_dir.exists():
        # Reuse the default espeak-ng-data if the downloaded voice folder doesn't have it.
        fallback = DEFAULT_MODEL_DIR / "espeak-ng-data"
        if fallback.exists():
            data_dir = fallback

    return model_path, tokens_path, data_dir


def create_tts(
    *,
    provider: str = "cpu",
    num_threads: int = 2,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
) -> sherpa_onnx.OfflineTts:
    model_dir = _resolve_model_dir()
    model_path, tokens_path, data_dir = _resolve_paths(model_dir)

    # Extra guard: some voices are trained with pinyin tokens and won't work with
    # sherpa-onnx's piper phonemizer.
    json_path = model_path.with_suffix(".onnx.json")
    if json_path.exists():
        try:
            import json

            obj = json.loads(json_path.read_text(encoding="utf-8"))
            phoneme_type = (obj.get("phoneme_type") or "").strip().lower()
            if phoneme_type == "pinyin":
                raise SystemExit(
                    "当前 Piper 音色为 phoneme_type=pinyin，无法在 sherpa-onnx 的 piper TTS 中使用。\n"
                    "建议：使用 zh_CN-huayan-*（espeak）或改用官方 piper 运行时。"
                )
        except SystemExit:
            raise
        except Exception:
            # Ignore config parse issues; sherpa-onnx will throw a clearer error later.
            pass

    if not model_path.exists():
        raise FileNotFoundError(f"Missing TTS model: {model_path}")
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Missing tokens.txt: {tokens_path}\n"
            "（你可以下载同目录的 *.onnx.json，让程序自动生成 tokens.txt）"
        )
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing espeak-ng-data: {data_dir}")

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model_path),
        tokens=str(tokens_path),
        data_dir=str(data_dir),
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

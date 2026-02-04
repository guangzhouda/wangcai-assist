import os
import time
from pathlib import Path


def _run_one(engine: str, text: str) -> None:
    out_dir = Path(__file__).resolve().parent / "output" / "tts_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{engine}.wav"

    if engine == "cosyvoice":
        from tts_cosyvoice import create_tts, synthesize_to_wav_with_duration

        tts = create_tts()
        t0 = time.perf_counter()
        _, dur = synthesize_to_wav_with_duration(tts, text, str(wav_path), speed=1.0)
        dt = time.perf_counter() - t0
        rtf = (dt / dur) if dur > 0 else float("inf")
        print(f"[cosyvoice] out={wav_path} dur={dur:.2f}s time={dt:.2f}s rtf={rtf:.3f}")
        return

    if engine == "openvoice":
        from tts_openvoice import create_tts, synthesize_to_wav_with_duration

        tts = create_tts()
        t0 = time.perf_counter()
        _, dur = synthesize_to_wav_with_duration(tts, text, str(wav_path), speed=1.0)
        dt = time.perf_counter() - t0
        rtf = (dt / dur) if dur > 0 else float("inf")
        print(f"[openvoice] out={wav_path} dur={dur:.2f}s time={dt:.2f}s rtf={rtf:.3f}")
        return

    if engine == "matcha":
        from tts_matcha import create_tts, synthesize_to_wav_with_duration

        tts = create_tts()
        t0 = time.perf_counter()
        _, dur = synthesize_to_wav_with_duration(tts, text, str(wav_path), speed=1.0)
        dt = time.perf_counter() - t0
        rtf = (dt / dur) if dur > 0 else float("inf")
        print(f"[matcha] out={wav_path} dur={dur:.2f}s time={dt:.2f}s rtf={rtf:.3f}")
        return

    if engine == "melo":
        from tts_melo import create_tts, synthesize_to_wav_with_duration

        tts = create_tts(provider="cpu")
        t0 = time.perf_counter()
        _, dur = synthesize_to_wav_with_duration(tts, text, str(wav_path), speed=1.0)
        dt = time.perf_counter() - t0
        rtf = (dt / dur) if dur > 0 else float("inf")
        print(f"[melo] out={wav_path} dur={dur:.2f}s time={dt:.2f}s rtf={rtf:.3f}")
        return

    raise ValueError(f"Unknown engine: {engine}")


def main() -> None:
    text = os.environ.get("TTS_COMPARE_TEXT", "").strip() or "你好，我是旺财。"
    engines = os.environ.get("TTS_COMPARE_ENGINES", "").strip()
    if engines:
        engine_list = [e.strip().lower() for e in engines.split(",") if e.strip()]
    else:
        engine_list = ["melo", "cosyvoice", "openvoice", "matcha"]

    print(f"Text: {text}")
    print(f"Engines: {engine_list}")
    for eng in engine_list:
        try:
            _run_one(eng, text)
        except SystemExit as exc:
            print(f"[{eng}] skipped: {exc}")
        except Exception as exc:
            print(f"[{eng}] failed: {exc}")


if __name__ == "__main__":
    main()

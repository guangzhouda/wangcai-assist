import os
import sys
import tempfile
import time
import threading
import logging
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

from llm_deepseek import stream_chat
from sherpa_asr import start_streaming_asr

TTS_ENGINE = os.environ.get("TTS_ENGINE", "piper").strip().lower()
if TTS_ENGINE == "cosyvoice":
    from tts_cosyvoice import create_tts, synthesize_to_wav_with_duration
else:
    from tts_piper import create_tts, synthesize_to_wav_with_duration


def play_wav_with_typewriter(
    wav_path: str,
    text: str,
    duration_sec: float,
    *,
    prefix: str = "",
    end: str = "",
) -> None:
    """Play a pre-synthesized wav while printing text at (roughly) audio pace."""
    try:
        import winsound

        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        # If playback fails, we still print the text.
        duration_sec = 0.0

    sys.stdout.write(prefix)
    sys.stdout.flush()

    if not text or duration_sec <= 0:
        sys.stdout.write(text + end)
        sys.stdout.flush()
        return

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

    # Ensure the audio has time to finish before returning.
    remaining = start + duration_sec - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def run_voice_chat_session(
    *,
    provider: Optional[str] = None,
    device_index: int = -1,
    stop_event: Optional[threading.Event] = None,
    handle_keyboard_interrupt: bool = True,
) -> None:
    """ASR -> LLM(stream) -> TTS(incremental).

    If stop_event is provided, the session will stop when it is set (e.g. when
    user says an exit phrase). If stop_event is None, an internal one is used.
    """
    if provider is None:
        provider = os.environ.get("ASR_PROVIDER", "cuda")

    if stop_event is None:
        stop_event = threading.Event()

    def quiet_logs() -> None:
        # Keep console output clean for voice chat. Set VOICE_DEBUG=1 to see logs.
        if os.environ.get("VOICE_DEBUG", "").strip().lower() not in ("1", "true", "yes"):
            root = logging.getLogger()
            root.setLevel(logging.WARNING)
            for name in (
                "httpx",
                "httpcore",
                "huggingface_hub",
                "transformers",
                "modelscope",
                "cosyvoice",
            ):
                logging.getLogger(name).setLevel(logging.WARNING)

    quiet_logs()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    trust_env = os.environ.get("DEEPSEEK_NO_PROXY", "").strip() not in ("1", "true", "yes")

    if TTS_ENGINE == "cosyvoice":
        tts = create_tts()
    else:
        tts = create_tts(provider="cpu")

    # Some third-party libs call logging.basicConfig() during import and reset the
    # level/handlers. Re-apply our console log policy after TTS init.
    quiet_logs()
    # warm-up: reduce the first synthesis latency
    try:
        _ = tts.generate("你好")
    except Exception:
        pass

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "你是一个中文语音助手，回答要简洁、自然。"},
    ]

    def _normalize_cmd(s: str) -> str:
        s = s.strip().lower().replace(" ", "")
        for ch in "，,。.!！？?;；:：\n\r\t":
            s = s.replace(ch, "")
        return s

    def _parse_exit_phrases(spec: str) -> set[str]:
        out: set[str] = set()
        for p in (spec or "").split(","):
            p = _normalize_cmd(p)
            if p:
                out.add(p)
        return out

    exit_phrases = _parse_exit_phrases(os.environ.get("VOICE_EXIT_PHRASES", "休眠,退出,结束,停止,再见,拜拜"))
    exit_ack = os.environ.get("VOICE_EXIT_ACK", "好的，我回到待机模式。")

    def split_ready_tts_chunks(
        buf: str,
        *,
        min_chars: int = 10,
        hard_max_chars: int = 36,
        allow_soft_cut: bool = False,
    ) -> tuple[List[str], str]:
        end_punct = set("。！？!?\\n")
        soft_punct = "，,;；:："
        soft_punct_set = set(soft_punct)

        chunks: List[str] = []
        while True:
            s = buf.lstrip()
            if len(s) < min_chars:
                buf = s
                break

            cut = None

            # Cut at the first sentence-ending punctuation after min_chars.
            for i, ch in enumerate(s):
                if ch in end_punct and (i + 1) >= min_chars:
                    cut = i + 1
                    break

            # For the first sentence, allow cutting earlier at soft punctuation
            # to reduce "first audio" latency.
            if cut is None and allow_soft_cut:
                for i, ch in enumerate(s):
                    if ch in soft_punct_set and (i + 1) >= min_chars:
                        cut = i + 1
                        break

            # If no sentence end yet, cut when it grows too long.
            if cut is None and len(s) >= hard_max_chars:
                window = s[:hard_max_chars]
                soft = -1
                for p in soft_punct:
                    soft = max(soft, window.rfind(p))
                if soft >= (min_chars - 1):
                    cut = soft + 1
                else:
                    cut = hard_max_chars

            if cut is None:
                buf = s
                break

            chunk = s[:cut].strip()
            if chunk:
                chunks.append(chunk)
            buf = s[cut:]

        return chunks, buf

    def on_final(user_text: str) -> None:
        nonlocal messages

        user_text = user_text.strip()
        if not user_text:
            return

        if _normalize_cmd(user_text) in exit_phrases:
            print(f"\n用户: {user_text}")
            # Speak a short ack and end this session.
            try:
                out_dir = Path(__file__).resolve().parent / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                fd, wav_path = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=str(out_dir))
                os.close(fd)
                wav_path, dur = synthesize_to_wav_with_duration(tts, exit_ack, wav_path, speed=1.0)
                play_wav_with_typewriter(wav_path, exit_ack, dur, prefix="助手: ", end="\n")
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            finally:
                stop_event.set()
            return

        print(f"\n用户: {user_text}")
        messages.append({"role": "user", "content": user_text})

        # Pipeline:
        # - LLM stream -> text chunks (fast)
        # - TTS synth (slow) in one worker thread (keeps model thread-safe)
        # - Playback/printing in another thread, so synthesis can overlap playback.
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        text_q: Queue[Optional[str]] = Queue()
        audio_q: Queue[Optional[Tuple[str, str, float]]] = Queue()

        def synth_worker() -> None:
            while True:
                part = text_q.get()
                try:
                    if part is None:
                        audio_q.put(None)
                        return

                    fd, wav_path = tempfile.mkstemp(prefix="tts_", suffix=".wav", dir=str(out_dir))
                    os.close(fd)

                    wav_path, dur = synthesize_to_wav_with_duration(
                        tts,
                        part,
                        wav_path,
                        speed=1.0,
                    )
                    audio_q.put((part, wav_path, float(dur)))
                finally:
                    text_q.task_done()

        def playback_worker() -> None:
            # Print assistant at speech pace by chunked TTS.
            print("助手: ", end="", flush=True)
            while True:
                item = audio_q.get()
                try:
                    if item is None:
                        return
                    part, wav_path, dur = item
                    play_wav_with_typewriter(wav_path, part, dur, prefix="", end="")
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
                finally:
                    audio_q.task_done()

        synth_t = threading.Thread(target=synth_worker, daemon=True)
        play_t = threading.Thread(target=playback_worker, daemon=True)
        synth_t.start()
        play_t.start()

        # First sentence strategy:
        # - Cut more aggressively at the beginning to get the first audio out fast.
        # - After we have started speaking, switch back to normal chunking for quality.
        first_min_chars = int(os.environ.get("TTS_FIRST_MIN_CHARS", "8"))
        first_hard_max = int(os.environ.get("TTS_FIRST_HARD_MAX_CHARS", "20"))
        first_deadline_sec = float(os.environ.get("TTS_FIRST_DEADLINE_SEC", "0.7"))
        first_force_min_chars = int(os.environ.get("TTS_FIRST_FORCE_MIN_CHARS", "4"))

        # Staircase chunking:
        # Start with shorter chunks (lower latency), then gradually allow longer chunks
        # to reduce synthesis overhead while audio is already playing.
        def parse_ladder(spec: str) -> List[Tuple[int, int]]:
            out: List[Tuple[int, int]] = []
            for part in (spec or "").split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    continue
                a, b = part.split(":", 1)
                try:
                    mn = int(a.strip())
                    mx = int(b.strip())
                except ValueError:
                    continue
                if mn <= 0 or mx <= 0:
                    continue
                if mx < mn:
                    mn, mx = mx, mn
                out.append((mn, mx))
            return out

        # After the first chunk, we ramp from short -> long.
        # You can override with: TTS_LADDER="10:24,12:32,14:40,16:56"
        ladder = parse_ladder(os.environ.get("TTS_LADDER", "10:28,12:36,14:48,16:64"))
        if not ladder:
            ladder = [(int(os.environ.get("TTS_MIN_CHARS", "10")), int(os.environ.get("TTS_HARD_MAX_CHARS", "36")))]

        soft_cut_steps = int(os.environ.get("TTS_SOFT_CUT_STEPS", "3"))
        enqueued_chunks = 0
        first_token_ts: Optional[float] = None
        started_speaking = False

        end_punct = set("。！？!?\n")
        soft_punct_set = set("，,;；:：")

        def force_first_chunk(cur_buf: str) -> Optional[Tuple[str, str]]:
            s = cur_buf.lstrip()
            if not s:
                return None
            if len(s) < first_force_min_chars:
                return None

            window = s[:first_hard_max]
            cut = None

            for i, ch in enumerate(window):
                if ch in end_punct:
                    cut = i + 1
                    break
            if cut is None:
                for i, ch in enumerate(window):
                    if ch in soft_punct_set:
                        cut = i + 1
                        break
            if cut is None:
                cut = len(window)

            part = s[:cut].strip()
            rest = s[cut:]
            if not part:
                return None
            return part, rest

        buf = ""
        reply_parts: List[str] = []
        try:
            for chunk in stream_chat(
                messages,
                api_key=api_key,
                model=model,
                trust_env=trust_env,
            ):
                if first_token_ts is None and chunk:
                    first_token_ts = time.perf_counter()

                reply_parts.append(chunk)
                buf += chunk

                if started_speaking:
                    stage = min(max(0, enqueued_chunks - 1), len(ladder) - 1)
                    cur_min, cur_max = ladder[stage]
                    allow_soft = enqueued_chunks < soft_cut_steps
                else:
                    cur_min, cur_max = first_min_chars, first_hard_max
                    allow_soft = True

                ready, buf = split_ready_tts_chunks(
                    buf,
                    min_chars=cur_min,
                    hard_max_chars=cur_max,
                    allow_soft_cut=allow_soft,
                )
                for part in ready:
                    text_q.put(part)
                    started_speaking = True
                    enqueued_chunks += 1

                if (
                    (not started_speaking)
                    and (first_token_ts is not None)
                    and (time.perf_counter() - first_token_ts) >= first_deadline_sec
                ):
                    forced = force_first_chunk(buf)
                    if forced is not None:
                        part, buf = forced
                        text_q.put(part)
                        started_speaking = True
                        enqueued_chunks += 1
        except Exception as exc:
            # Stop workers and show error on a new line.
            text_q.put(None)
            text_q.join()
            audio_q.join()
            print(f"\n[LLM error] {exc}")
            return

        rest = buf.strip()
        if rest:
            text_q.put(rest)

        # Signal end and wait for all chunks to be synthesized and spoken.
        text_q.put(None)
        text_q.join()
        audio_q.join()
        print()  # newline after assistant reply

        reply = "".join(reply_parts).strip()
        if reply:
            messages.append({"role": "assistant", "content": reply})

        # 控制上下文长度，避免越聊越长
        if len(messages) > 20:
            messages = messages[:1] + messages[-18:]

    start_streaming_asr(
        provider=provider,
        device_index=device_index,
        on_final=on_final,
        stop_event=stop_event,
        handle_keyboard_interrupt=handle_keyboard_interrupt,
    )


def main() -> None:
    run_voice_chat_session()


if __name__ == "__main__":
    main()

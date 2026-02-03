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
elif TTS_ENGINE == "openvoice":
    from tts_openvoice import create_tts, synthesize_to_wav_with_duration
elif TTS_ENGINE in ("piper_native", "piper-official", "piper_cli"):
    from tts_piper_native import create_tts, synthesize_to_wav_with_duration
else:
    from tts_piper import create_tts, synthesize_to_wav_with_duration


def init_tts_from_env():
    """Initialize a TTS instance once and reuse it across wake sessions.

    This avoids re-loading large models (e.g. CosyVoice) after every wakeword.
    """
    if TTS_ENGINE == "cosyvoice":
        tts = create_tts()
    elif TTS_ENGINE == "openvoice":
        tts = create_tts()
    elif TTS_ENGINE in ("piper_native", "piper-official", "piper_cli"):
        tts = create_tts()
    else:
        tts = create_tts(provider="cpu")

    # Warm-up: reduce first synthesis latency.
    try:
        _ = tts.generate("你好")
    except Exception:
        pass

    return tts


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
    tts_instance=None,
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

    if tts_instance is not None:
        tts = tts_instance
    else:
        if TTS_ENGINE == "cosyvoice":
            tts = create_tts()
        elif TTS_ENGINE == "openvoice":
            tts = create_tts()
        elif TTS_ENGINE in ("piper_native", "piper-official", "piper_cli"):
            tts = create_tts()
        else:
            tts = create_tts(provider="cpu")

    # Some third-party libs call logging.basicConfig() during import and reset the
    # level/handlers. Re-apply our console log policy after TTS init.
    quiet_logs()
    # warm-up: reduce the first synthesis latency (only when we created the instance)
    if tts_instance is None:
        try:
            _ = tts.generate("你好")
        except Exception:
            pass

    # Optional: pre-synthesize a very short "thinking" ack to reduce perceived latency
    # for the first audio (especially when LLM first-token or TTS is slow).
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    think_ack_text = os.environ.get("VOICE_THINK_ACK_TEXT", "嗯").strip()
    think_ack_delay = float(os.environ.get("VOICE_THINK_ACK_DELAY", "0.6"))
    think_ack_wav: Optional[str] = None
    if think_ack_text:
        try:
            ack_path = out_dir / f"think_ack_{TTS_ENGINE}.wav"
            wav_path, _dur = synthesize_to_wav_with_duration(tts, think_ack_text, str(ack_path), speed=1.0)
            think_ack_wav = wav_path
        except Exception:
            think_ack_wav = None

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
        # "Soft" cut points (less ideal than sentence end, but better than cutting mid-word).
        soft_punct = "，,;；:：、)]）】》"
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
        # NOTE: out_dir is created once above and reused here.

        text_q: Queue[Optional[str]] = Queue()
        audio_q: Queue[Optional[Tuple[str, str, float]]] = Queue()

        # Ensure we don't overlap different wav playbacks (pre-ack vs answer chunks).
        play_lock = threading.Lock()
        started_playback = threading.Event()

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

            # Optional jitter buffer: wait for a bit of audio to be ready before we start
            # playing, so chunk boundaries are less likely to produce audible gaps.
            prebuffer_sec = float(os.environ.get("TTS_PREBUFFER_SEC", "0.4"))
            buffered: List[Tuple[str, str, float]] = []
            buffered_dur = 0.0
            started = False

            while True:
                item = audio_q.get()
                if item is None:
                    # Drain buffered items before exit.
                    while buffered:
                        part0, wav0, dur0 = buffered.pop(0)
                        buffered_dur = max(0.0, buffered_dur - float(dur0))
                        if not started_playback.is_set():
                            started_playback.set()
                        with play_lock:
                            play_wav_with_typewriter(wav0, part0, dur0, prefix="", end="")
                        try:
                            os.remove(wav0)
                        except Exception:
                            pass
                        audio_q.task_done()

                    audio_q.task_done()  # ack the sentinel
                    return

                part, wav_path, dur = item
                buffered.append((part, wav_path, dur))
                buffered_dur += float(dur)

                if (not started) and prebuffer_sec > 0 and buffered_dur < prebuffer_sec:
                    # Keep buffering.
                    continue

                started = True
                # Play as many buffered items as possible (we keep buffering during playback
                # via the TTS worker thread).
                while buffered:
                    part0, wav0, dur0 = buffered.pop(0)
                    buffered_dur = max(0.0, buffered_dur - float(dur0))
                    if not started_playback.is_set():
                        started_playback.set()
                    with play_lock:
                        play_wav_with_typewriter(wav0, part0, dur0, prefix="", end="")
                    try:
                        os.remove(wav0)
                    except Exception:
                        pass
                    audio_q.task_done()

        synth_t = threading.Thread(target=synth_worker, daemon=True)
        play_t = threading.Thread(target=playback_worker, daemon=True)
        synth_t.start()
        play_t.start()

        # Play a short "thinking" ack if we are still not speaking after a short delay.
        # This hides LLM first-token latency and the first TTS chunk synth time.
        if think_ack_wav and think_ack_delay > 0:
            def _ack_worker() -> None:
                try:
                    time.sleep(think_ack_delay)
                    if started_playback.is_set():
                        return
                    if not play_lock.acquire(blocking=False):
                        return
                    try:
                        if started_playback.is_set():
                            return
                        import winsound

                        winsound.PlaySound(think_ack_wav, winsound.SND_FILENAME)
                    finally:
                        play_lock.release()
                except Exception:
                    pass

            threading.Thread(target=_ack_worker, daemon=True).start()

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
        # Default is tuned for smoother audio on local TTS (avoid cutting words too often).
        # You can override with: TTS_LADDER="10:32,12:44,14:60,16:80"
        ladder = parse_ladder(os.environ.get("TTS_LADDER", "10:32,12:44,14:60,16:80"))
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

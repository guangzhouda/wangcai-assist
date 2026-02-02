import os
import threading

from voice_chat import init_tts_from_env, run_voice_chat_session
from wakeword import get_access_key_from_env, wait_for_wakeword


def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def main() -> None:
    access_key = get_access_key_from_env()
    if not access_key:
        raise SystemExit(
            "找不到 PICOVOICE_ACCESS_KEY。\n"
            "可选方案：\n"
            "1) 设置环境变量 PICOVOICE_ACCESS_KEY\n"
            "2) 或把 key 写入 model/旺财_zh_windows_v4_0_0/LICENSE.txt（本仓库会自动读取）"
        )

    # Mic device selection:
    # - KWS 和 ASR 默认用同一个麦克风索引（-1 为系统默认）
    mic_device_index = _get_int_env("MIC_DEVICE_INDEX", -1)
    kws_device_index = _get_int_env("KWS_DEVICE_INDEX", mic_device_index)

    kws_sensitivity = _get_float_env("KWS_SENSITIVITY", 0.5)
    asr_provider = os.environ.get("ASR_PROVIDER", "cuda").strip() or "cuda"

    # Preload TTS in background so the first wake doesn't pay the cold-start cost
    # (CosyVoice/OpenVoice can take a while to load or download text frontend models).
    tts_ref: dict[str, object] = {"tts": None, "err": None}
    tts_ready = threading.Event()

    def _load_tts() -> None:
        try:
            tts_ref["tts"] = init_tts_from_env()
        except Exception as exc:  # keep main loop alive
            tts_ref["err"] = exc
        finally:
            tts_ready.set()

    threading.Thread(target=_load_tts, daemon=True).start()

    print("待机中：说“旺财”唤醒，Ctrl+C 退出。")
    print("提示：唤醒后说“休眠/退出/再见”可回到待机。")

    while True:
        try:
            ok = wait_for_wakeword(
                access_key=access_key,
                device_index=kws_device_index,
                sensitivity=kws_sensitivity,
            )
        except KeyboardInterrupt:
            print("\n已退出。")
            return

        if not ok:
            # timeout / stop_event（当前没有传）才会到这里；预留。
            continue

        print("✅ 已唤醒：旺财")
        if not tts_ready.is_set():
            print("⏳ 正在加载语音合成模型，请稍等...")
            tts_ready.wait()
        if tts_ref.get("err") is not None:
            raise SystemExit(f"TTS 初始化失败：{tts_ref['err']}")

        stop_event = threading.Event()
        try:
            run_voice_chat_session(
                provider=asr_provider,
                device_index=mic_device_index,
                stop_event=stop_event,
                tts_instance=tts_ref.get("tts"),
                # 在整合模式下，让 Ctrl+C 直接退出整个程序。
                handle_keyboard_interrupt=False,
            )
        except KeyboardInterrupt:
            print("\n已退出。")
            return

        print("🟡 已回到待机，继续监听唤醒词...")


if __name__ == "__main__":
    main()

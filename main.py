import os
import threading

from voice_chat import run_voice_chat_session
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
        stop_event = threading.Event()
        try:
            run_voice_chat_session(
                provider=asr_provider,
                device_index=mic_device_index,
                stop_event=stop_event,
                # 在整合模式下，让 Ctrl+C 直接退出整个程序。
                handle_keyboard_interrupt=False,
            )
        except KeyboardInterrupt:
            print("\n已退出。")
            return

        print("🟡 已回到待机，继续监听唤醒词...")


if __name__ == "__main__":
    main()

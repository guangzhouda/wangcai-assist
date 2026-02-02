import threading

from silero_vad import start_vad_listener
from wakeword import get_access_key_from_env, start_wakeword_listener


def main() -> None:
    vad_started = False

    def start_vad_thread() -> None:
        def on_speech_start() -> None:
            print("🎤 VAD: 检测到语音开始")

        def on_speech_end() -> None:
            print("🛑 VAD: 检测到语音结束")

        start_vad_listener(
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
        )

    def on_wake() -> None:
        nonlocal vad_started
        print("✅ 已唤醒：旺财")
        if not vad_started:
            vad_started = True
            print("启动 VAD 监听...")
            thread = threading.Thread(target=start_vad_thread, daemon=True)
            thread.start()

    access_key = get_access_key_from_env()
    if not access_key:
        raise SystemExit("请设置环境变量 PICOVOICE_ACCESS_KEY")

    print("开始监听唤醒词，按 Ctrl+C 退出...")
    start_wakeword_listener(access_key=access_key, on_wake=on_wake)


if __name__ == "__main__":
    main()

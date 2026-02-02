import os
from pathlib import Path
from typing import Callable, List, Tuple

import pvporcupine

try:
    from pvrecorder import PvRecorder
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 pvrecorder。请先安装：pip install pvrecorder"
    ) from exc


def resolve_default_paths() -> Tuple[str, str]:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model" / "旺财_zh_windows_v4_0_0"
    keyword_path = model_dir / "旺财_zh_windows_v4_0_0.ppn"
    model_path = model_dir / "porcupine_params_zh.pv"
    return str(keyword_path), str(model_path)


def get_available_devices() -> List[str]:
    return PvRecorder.get_available_devices()


def start_wakeword_listener(
    access_key: str,
    keyword_path: str = "",
    model_path: str = "",
    device_index: int = -1,
    sensitivity: float = 0.5,
    on_wake: Callable[[], None] | None = None,
) -> None:
    if not access_key:
        raise ValueError("access_key 不能为空")

    default_keyword_path, default_model_path = resolve_default_paths()
    keyword_path = keyword_path or default_keyword_path
    model_path = model_path or default_model_path

    if not Path(keyword_path).exists():
        raise FileNotFoundError(f"找不到唤醒词模型文件：{keyword_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"找不到语言模型文件：{model_path}")

    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[keyword_path],
        model_path=model_path,
        sensitivities=[sensitivity],
    )

    recorder = PvRecorder(
        device_index=device_index,
        frame_length=porcupine.frame_length,
    )

    if on_wake is None:
        def on_wake_default() -> None:
            print("✅ 已唤醒：旺财")

        on_wake = on_wake_default

    try:
        recorder.start()
        while True:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                on_wake()
    finally:
        if recorder.is_recording:
            recorder.stop()
        recorder.delete()
        porcupine.delete()


def get_access_key_from_env() -> str:
    return os.environ.get("PICOVOICE_ACCESS_KEY", "")

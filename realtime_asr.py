import threading
from collections import deque
from queue import Queue
from typing import Deque, List

import numpy as np
from faster_whisper import WhisperModel

from silero_vad import SileroVADOnnx, VADConfig, resolve_default_model_path

try:
    from pvrecorder import PvRecorder
except ImportError as exc:
    raise SystemExit("缺少依赖 pvrecorder。请先安装：pip install pvrecorder") from exc


MODEL_DIR = r"E:\Projects\wangcai-assist\model\large-v3-turbo"
DEVICE = "cuda"  # 没有 GPU 可改成 "cpu"
COMPUTE_TYPE = "int8_float16"
LANGUAGE = "zh"


def _pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def _build_model() -> WhisperModel:
    try:
        return WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception:
        # GPU 不可用时回退到 CPU
        return WhisperModel(MODEL_DIR, device="cpu", compute_type="int8")


def _asr_worker(q: Queue) -> None:
    model = _build_model()
    while True:
        audio = q.get()
        if audio is None:
            break
        segments, _info = model.transcribe(audio, language=LANGUAGE)
        text = "".join(seg.text for seg in segments).strip()
        if text:
            print(f"识别: {text}")
        q.task_done()


def main() -> None:
    cfg = VADConfig()
    vad = SileroVADOnnx(onnx_path=resolve_default_model_path(), cfg=cfg)

    recorder = PvRecorder(device_index=-1, frame_length=cfg.window)

    pad_frames = max(1, int(cfg.speech_pad_ms / 1000 * cfg.sr / cfg.window))
    pre_speech: Deque[np.ndarray] = deque(maxlen=pad_frames)
    speech_frames: List[np.ndarray] = []

    q: Queue = Queue()
    worker = threading.Thread(target=_asr_worker, args=(q,), daemon=True)
    worker.start()

    print("开始实时中文识别，按 Ctrl+C 退出...")

    try:
        recorder.start()
        while True:
            pcm = np.array(recorder.read(), dtype=np.int16)
            x = _pcm16_to_float32(pcm)
            pre_speech.append(x)

            speech_prob = vad.step(x)
            event = vad.update_endpoint(speech_prob)

            if vad.in_speech:
                speech_frames.append(x)

            if event == "start":
                speech_frames = list(pre_speech)
            elif event == "end":
                if speech_frames:
                    audio = np.concatenate(speech_frames, axis=0)
                    q.put(audio)
                speech_frames = []
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        if recorder.is_recording:
            recorder.stop()
        recorder.delete()
        q.put(None)


if __name__ == "__main__":
    main()

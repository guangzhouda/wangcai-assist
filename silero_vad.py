import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import onnxruntime as ort

try:
    from pvrecorder import PvRecorder
except ImportError as exc:
    raise SystemExit(
        "缺少依赖 pvrecorder。请先安装：pip install pvrecorder"
    ) from exc

@dataclass
class VADConfig:
    sr: int = 16000
    window: int = 512              # 32ms @16k
    threshold: float = 0.5         # speech_prob >= threshold 认为是语音
    min_silence_ms: int = 300      # 静音超过该时长，判定一句话结束
    speech_pad_ms: int = 80        # 句首句尾补偿（你做切段时用）

class SileroVADOnnx:
    def __init__(self, onnx_path: str, cfg: VADConfig = VADConfig(), providers=None):
        self.cfg = cfg
        if providers is None:
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        # 兼容不同版本：动态查输入名
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]

        # 常见输入：input, sr, h, c
        # 常见输出：out(=speech_prob), h, c
        self._reset_states()

        self.in_speech = False
        self.silence_ms = 0

    def _reset_states(self):
        # 通过 session input shape 推断 state 形状（兼容 state 或 h/c）
        state_name = next((n for n in self.in_names if n.lower() == "state"), None)
        h_name = next((n for n in self.in_names if n.lower() == "h"), None)
        c_name = next((n for n in self.in_names if n.lower() == "c"), None)

        if state_name:
            state_shape = self.sess.get_inputs()[self.in_names.index(state_name)].shape
            state_shape = [
                1 if (s is None or isinstance(s, str)) else int(s) for s in state_shape
            ]
            self.state = np.zeros(state_shape, dtype=np.float32)
            self.h, self.c = None, None
        elif h_name and c_name:
            h_shape = self.sess.get_inputs()[self.in_names.index(h_name)].shape
            c_shape = self.sess.get_inputs()[self.in_names.index(c_name)].shape
            h_shape = [1 if (s is None or isinstance(s, str)) else int(s) for s in h_shape]
            c_shape = [1 if (s is None or isinstance(s, str)) else int(s) for s in c_shape]
            self.h = np.zeros(h_shape, dtype=np.float32)
            self.c = np.zeros(c_shape, dtype=np.float32)
            self.state = None
        else:
            self.h, self.c, self.state = None, None, None

    def reset(self):
        self._reset_states()
        self.in_speech = False
        self.silence_ms = 0

    def _make_inputs(self, x_512: np.ndarray):
        # x_512: shape (512,) float32, [-1,1]
        x = x_512.astype(np.float32)[None, :]  # (1, 512)

        feed = {}
        for n in self.in_names:
            nl = n.lower()
            if nl in ("input", "x", "audio"):
                feed[n] = x
            elif nl in ("sr", "sampling_rate"):
                feed[n] = np.array([self.cfg.sr], dtype=np.int64)
            elif nl == "state" and self.state is not None:
                feed[n] = self.state
            elif nl == "h" and self.h is not None:
                feed[n] = self.h
            elif nl == "c" and self.c is not None:
                feed[n] = self.c
        return feed

    def step(self, x_512: np.ndarray) -> float:
        feed = self._make_inputs(x_512)
        outs = self.sess.run(None, feed)

        # 经验：第一个输出一般是 speech_prob，后面是 state 或 h/c
        speech_prob = float(np.squeeze(outs[0]))
        if self.state is not None and len(outs) >= 2:
            self.state = outs[1]
        elif self.h is not None and self.c is not None and len(outs) >= 3:
            self.h = outs[1]
            self.c = outs[2]
        return speech_prob

    def update_endpoint(self, speech_prob: float):
        """返回事件：None / 'start' / 'end'"""
        win_ms = int(self.cfg.window / self.cfg.sr * 1000)

        if speech_prob >= self.cfg.threshold:
            self.silence_ms = 0
            if not self.in_speech:
                self.in_speech = True
                return "start"
            return None
        else:
            if self.in_speech:
                self.silence_ms += win_ms
                if self.silence_ms >= self.cfg.min_silence_ms:
                    self.in_speech = False
                    self.silence_ms = 0
                    return "end"
            return None


def resolve_default_model_path() -> str:
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model" / "silero_vad" / "silero_vad.onnx"
    return str(model_path)


def get_available_devices() -> List[str]:
    return PvRecorder.get_available_devices()


def _pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def start_vad_listener(
    onnx_path: str = "",
    cfg: VADConfig = VADConfig(),
    device_index: int = -1,
    on_speech_start: Callable[[], None] | None = None,
    on_speech_end: Callable[[], None] | None = None,
    on_vad_prob: Callable[[float], None] | None = None,
) -> None:
    if not onnx_path:
        onnx_path = resolve_default_model_path()

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"找不到 VAD 模型文件：{onnx_path}")

    if cfg.window != 512:
        raise ValueError("Silero VAD 目前仅支持 512 采样窗口（32ms @16k）")

    vad = SileroVADOnnx(onnx_path=onnx_path, cfg=cfg)
    recorder = PvRecorder(device_index=device_index, frame_length=cfg.window)

    try:
        recorder.start()
        while True:
            pcm = np.array(recorder.read(), dtype=np.int16)
            x = _pcm16_to_float32(pcm)
            speech_prob = vad.step(x)
            if on_vad_prob is not None:
                on_vad_prob(speech_prob)

            event = vad.update_endpoint(speech_prob)
            if event == "start" and on_speech_start is not None:
                on_speech_start()
            elif event == "end" and on_speech_end is not None:
                on_speech_end()
    finally:
        if recorder.is_recording:
            recorder.stop()
        recorder.delete()

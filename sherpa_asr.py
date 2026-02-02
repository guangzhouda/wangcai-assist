import sys
from pathlib import Path
from collections import deque
from typing import Callable, Optional

import numpy as np
import sherpa_onnx

from silero_vad import SileroVADOnnx, VADConfig, resolve_default_model_path

try:
    from pvrecorder import PvRecorder
except ImportError as exc:
    raise SystemExit("缺少依赖 pvrecorder。请先安装：pip install pvrecorder") from exc


MODEL_DIR = r"E:\Projects\wangcai-assist\model\sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
SAMPLE_RATE = 16000
FRAME_LENGTH = 512

# VAD 只负责切句与提交；pre-roll 用于补齐说话开头
VAD_ONNX_PATH = resolve_default_model_path()
VAD_THRESHOLD = 0.35
VAD_MIN_SILENCE_MS = 500
PRE_ROLL_MS = 200


def _pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def create_recognizer(model_dir: str, provider: str = "cpu") -> sherpa_onnx.OnlineRecognizer:
    model_path = Path(model_dir)
    tokens = model_path / "tokens.txt"
    encoder = model_path / "encoder-epoch-99-avg-1.onnx"
    decoder = model_path / "decoder-epoch-99-avg-1.onnx"
    joiner = model_path / "joiner-epoch-99-avg-1.onnx"

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens),
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        provider=provider,
        # 端点交给 Silero VAD 控制，避免过早 cut
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
        model_type="zipformer",
    )


def start_streaming_asr(
    model_dir: str = MODEL_DIR,
    provider: str = "cpu",
    device_index: int = -1,
    on_final: Callable[[str], None] | None = None,
    pause_mic_during_on_final: bool = True,
) -> None:
    recognizer = create_recognizer(model_dir=model_dir, provider=provider)
    recorder = PvRecorder(device_index=device_index, frame_length=FRAME_LENGTH)

    print("开始 Sherpa-ONNX 实时识别（中英混合），按 Ctrl+C 退出...")

    vad_cfg = VADConfig(
        threshold=VAD_THRESHOLD,
        min_silence_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=PRE_ROLL_MS,
        window=FRAME_LENGTH,
        sr=SAMPLE_RATE,
    )
    vad = SileroVADOnnx(onnx_path=VAD_ONNX_PATH, cfg=vad_cfg)

    frame_ms = int(FRAME_LENGTH / SAMPLE_RATE * 1000)
    pre_roll_frames = max(1, int(np.ceil(PRE_ROLL_MS / max(1, frame_ms))))
    pre_roll = deque(maxlen=pre_roll_frames)

    in_utt = False
    stream = None
    last_text: Optional[str] = None
    last_display_len = 0

    try:
        recorder.start()
        while True:
            pcm = np.array(recorder.read(), dtype=np.int16)
            samples = _pcm16_to_float32(pcm)

            speech_prob = vad.step(samples)
            event = vad.update_endpoint(speech_prob)

            if not in_utt:
                pre_roll.append(samples)

            if event == "start":
                in_utt = True
                stream = recognizer.create_stream()

                # 把说话开始前的 pre-roll 音频一起喂入，减少漏前半句
                if pre_roll:
                    stream.accept_waveform(SAMPLE_RATE, np.concatenate(list(pre_roll), axis=0))
                pre_roll.clear()
                last_text = None
                continue

            if in_utt and stream is not None:
                stream.accept_waveform(SAMPLE_RATE, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                text = recognizer.get_result(stream)
                if text and text != last_text:
                    # In-place update: keep partial results on a single line.
                    line = f"部分: {text}"
                    sys.stdout.write("\r" + line + (" " * max(0, last_display_len - len(line))))
                    sys.stdout.flush()
                    last_display_len = len(line)
                    last_text = text

            if event == "end" and in_utt:
                # Clear the partial line before printing final output.
                if last_display_len:
                    sys.stdout.write("\r" + (" " * last_display_len) + "\r")
                    sys.stdout.flush()
                    last_display_len = 0
                if last_text:
                    print(f"最终: {last_text}")

                    if on_final is not None:
                        # Avoid capturing the speaker playback as a new utterance.
                        if pause_mic_during_on_final and recorder.is_recording:
                            recorder.stop()
                        try:
                            on_final(last_text)
                        except Exception as exc:
                            print(f"[on_final error] {exc}")
                        finally:
                            if pause_mic_during_on_final and not recorder.is_recording:
                                recorder.start()
                            vad.reset()
                            pre_roll.clear()
                in_utt = False
                stream = None
                last_text = None
                pre_roll.clear()
    except KeyboardInterrupt:
        # Ensure we start from a new line even if we are updating in-place.
        if last_display_len:
            sys.stdout.write("\r" + (" " * last_display_len) + "\r")
            sys.stdout.flush()
        print("\n已停止。")
    finally:
        if recorder.is_recording:
            recorder.stop()
        recorder.delete()


if __name__ == "__main__":
    start_streaming_asr(provider="cuda")

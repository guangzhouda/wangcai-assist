import os
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# Default to an existing Piper voice folder (gitignored). You can override via env vars.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model" / "vits-piper-zh_CN-huayan-medium"


def _resolve_model_dir() -> Path:
    # Prefer a dedicated env var for the official Piper runtime backend.
    for k in ("PIPER_NATIVE_MODEL_DIR", "PIPER_MODEL_DIR"):
        v = os.environ.get(k, "").strip()
        if v:
            return Path(v).expanduser()
    return DEFAULT_MODEL_DIR


def _resolve_piper_bin() -> Path:
    v = os.environ.get("PIPER_BIN", "").strip()
    if v:
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent / p).resolve()
        return p

    base = Path(__file__).resolve().parent
    candidates = [
        base / "third_party" / "piper" / "piper.exe",
        base / "third_party" / "piper" / "piper",
        base / "piper.exe",
        base / "piper",
    ]
    for c in candidates:
        if c.exists():
            return c

    which = shutil.which("piper.exe") or shutil.which("piper")
    if which:
        return Path(which)

    raise FileNotFoundError(
        "找不到官方 Piper 运行时（piper / piper.exe）。\n"
        "请下载 Piper release，把 piper.exe 放到：third_party/piper/piper.exe，或设置环境变量 PIPER_BIN 指向它。"
    )


def _resolve_voice_and_config(model_dir: Path) -> Tuple[Path, Optional[Path]]:
    model_dir = model_dir.expanduser()

    model_path_env = (os.environ.get("PIPER_NATIVE_ONNX", "") or os.environ.get("PIPER_ONNX", "")).strip()
    if model_path_env:
        p = Path(model_path_env).expanduser()
        if not p.is_absolute():
            p = (model_dir / p).resolve()
        model_path = p
    else:
        onnx_files = sorted([p for p in model_dir.glob("*.onnx") if p.is_file()])
        if not onnx_files:
            onnx_files = sorted(
                [
                    p
                    for p in model_dir.rglob("*.onnx")
                    if p.is_file() and ".cache" not in p.parts and ".git" not in p.parts
                ]
            )
        if len(onnx_files) != 1:
            raise FileNotFoundError(
                f"Missing/ambiguous *.onnx in: {model_dir} (found {len(onnx_files)})\n"
                "请设置环境变量 PIPER_NATIVE_ONNX 指向具体的 .onnx 文件。"
            )
        model_path = onnx_files[0]

    cfg_env = (os.environ.get("PIPER_NATIVE_CONFIG", "") or os.environ.get("PIPER_CONFIG", "")).strip()
    if cfg_env:
        cfg = Path(cfg_env).expanduser()
        if not cfg.is_absolute():
            cfg = (model_dir / cfg).resolve()
        config_path = cfg
    else:
        # Typical: model.onnx + model.onnx.json
        cfg = model_path.with_suffix(model_path.suffix + ".json")
        config_path = cfg if cfg.exists() else None

    return model_path, config_path


def _wav_duration_sec(wav_path: str) -> float:
    try:
        with wave.open(wav_path, "rb") as w:
            fr = float(w.getframerate() or 0)
            n = float(w.getnframes() or 0)
            if fr <= 0:
                return 0.0
            return n / fr
    except Exception:
        return 0.0


@dataclass
class PiperNativeTTS:
    piper_bin: Path
    model_path: Path
    config_path: Optional[Path] = None
    speaker: Optional[int] = None
    _supported: set[str] = field(default_factory=set, init=False, repr=False)
    _probed: bool = field(default=False, init=False, repr=False)

    def _probe(self) -> None:
        if self._probed:
            return
        self._probed = True
        try:
            proc = subprocess.run(
                [str(self.piper_bin), "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for flag in ("--config", "--speaker", "--length_scale", "--output_file"):
                if flag in out:
                    self._supported.add(flag)
        except Exception:
            # If probing fails, we will still try a conservative command later.
            self._supported = set()

    def synthesize_to_wav(self, text: str, wav_path: str, *, speed: float = 1.0) -> str:
        self._probe()

        text = (text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        cmd = [str(self.piper_bin), "--model", str(self.model_path)]

        # Some builds accept --config, some auto-discover model.onnx.json next to model.
        if self.config_path and "--config" in self._supported:
            cmd += ["--config", str(self.config_path)]

        if self.speaker is not None and "--speaker" in self._supported:
            cmd += ["--speaker", str(int(self.speaker))]

        # Piper's length_scale: >1 slower, <1 faster. Map speed (~x) -> length_scale (~1/x).
        if speed and speed > 0 and "--length_scale" in self._supported:
            length_scale = 1.0 / float(speed)
            cmd += ["--length_scale", f"{length_scale:.4f}"]

        if "--output_file" in self._supported:
            cmd += ["--output_file", str(wav_path)]
        else:
            # Most releases support --output_file. If not, fail with a clear error.
            raise RuntimeError(
                "当前 Piper 运行时不支持 --output_file 参数，无法写入 wav。\n"
                "请确认你下载的是官方 piper 可执行文件（而不是其他同名工具）。"
            )

        proc = subprocess.run(
            cmd,
            input=text + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            tail = ((proc.stderr or "").strip() or (proc.stdout or "").strip())[-1200:]
            raise RuntimeError(f"Piper 合成失败（exit={proc.returncode}）。\n{tail}")

        if not Path(wav_path).exists():
            raise RuntimeError("Piper 运行结束但未生成 wav 文件，请检查 piper 输出/模型路径。")

        return wav_path


def create_tts(*, speaker: Optional[int] = None) -> PiperNativeTTS:
    """Create a Piper TTS wrapper that calls the official piper runtime.

    This backend is intended for voices that are NOT compatible with sherpa-onnx's
    piper phonemizer (e.g. phoneme_type=pinyin voices like xiao_ya/chaowen).
    """
    model_dir = _resolve_model_dir()
    piper_bin = _resolve_piper_bin()
    model_path, config_path = _resolve_voice_and_config(model_dir)

    if speaker is None:
        v = os.environ.get("PIPER_SPEAKER", "").strip()
        if v:
            try:
                speaker = int(v)
            except ValueError:
                speaker = None

    if not piper_bin.exists():
        raise FileNotFoundError(f"Missing piper binary: {piper_bin}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Piper voice model: {model_path}")
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Missing Piper config: {config_path}")

    return PiperNativeTTS(
        piper_bin=piper_bin,
        model_path=model_path,
        config_path=config_path,
        speaker=speaker,
    )


def synthesize_to_wav_with_duration(
    tts: PiperNativeTTS,
    text: str,
    wav_path: str,
    *,
    speed: float = 1.0,
) -> tuple[str, float]:
    wav_path = tts.synthesize_to_wav(text, wav_path, speed=speed)
    return wav_path, _wav_duration_sec(wav_path)


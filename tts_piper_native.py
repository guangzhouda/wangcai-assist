import json
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
        if c.exists() and c.is_file():
            return c

    # Some zip layouts contain a nested folder (e.g. third_party/piper/piper/piper.exe).
    nested_root = base / "third_party" / "piper"
    if nested_root.exists():
        try:
            exe = next(iter(nested_root.rglob("piper.exe")), None)
            if exe and exe.exists():
                return exe
        except Exception:
            pass

    which = shutil.which("piper.exe") or shutil.which("piper")
    if which:
        return Path(which)

    raise FileNotFoundError(
        "找不到官方 Piper 运行时（piper / piper.exe）。\n"
        "请下载 Piper release，把 piper.exe 放到：third_party/piper/piper.exe，或设置环境变量 PIPER_BIN 指向它。"
    )


def _resolve_voice_and_config(model_dir: Path) -> Tuple[Path, Optional[Path]]:
    model_dir = model_dir.expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"找不到 Piper 音色目录：{model_dir}\n"
            "请把音色（*.onnx + *.onnx.json）放到该目录，或设置环境变量 PIPER_NATIVE_MODEL_DIR 指向正确目录。"
        )

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


def _is_single_codepoint(s: str) -> bool:
    # Piper config expects each phoneme key to be exactly one unicode codepoint.
    return len(s) == 1


def _safe_int(v: object, default: int) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _load_config_or_none(path: Optional[Path]) -> Optional[dict]:
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_inference_scales(cfg: dict, *, speed: float = 1.0) -> Tuple[float, float, float]:
    inf = cfg.get("inference") or {}
    noise_scale = float(inf.get("noise_scale", 0.667))
    base_length_scale = float(inf.get("length_scale", 1.0))
    noise_w = float(inf.get("noise_w", 0.8))

    if speed and speed > 0:
        length_scale = base_length_scale / float(speed)
    else:
        length_scale = base_length_scale

    return noise_scale, length_scale, noise_w


def _pinyin_text_to_tokens(text: str, *, phoneme_id_map: Dict[str, List[int]]) -> List[str]:
    """Convert Chinese text -> tokens compatible with piper pinyin voices.

    The pinyin voices in this repo ship a phoneme_id_map with multi-character tokens
    (e.g. "zh", "ai", "uang"). The official piper.exe cannot load such configs.

    We do the tokenization ourselves using pypinyin:
    - 每个汉字 -> pinyin(TONE3) 例如 jin1
    - 拆成: 声母(或 Ø) + 韵母 + 声调数字
    - 标点/空格：如果在 phoneme_id_map 里，原样输出
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        from pypinyin import Style, pinyin  # type: ignore
    except Exception as exc:
        raise RuntimeError("缺少依赖 pypinyin，无法对 pinyin 音色做文本转音素。请先 pip install pypinyin") from exc

    initials = [
        "zh",
        "ch",
        "sh",
        "b",
        "p",
        "m",
        "f",
        "d",
        "t",
        "n",
        "l",
        "g",
        "k",
        "h",
        "j",
        "q",
        "x",
        "r",
        "z",
        "c",
        "s",
        "y",
        "w",
    ]

    out: List[str] = []
    seq = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    for item in seq:
        py = (item[0] or "").strip()
        if not py:
            continue

        # Pass through punctuation / spaces when supported.
        if py in phoneme_id_map and _is_single_codepoint(py):
            out.append(py)
            continue

        # Unknown non-Chinese chunk (e.g. "hello") -> map to space if possible.
        if py not in phoneme_id_map and not py[-1:].isdigit():
            if " " in phoneme_id_map:
                out.append(" ")
            continue

        # Parse tone digit.
        tone = py[-1] if py[-1:].isdigit() else "5"
        base = py[:-1] if py[-1:].isdigit() else py

        # Split initial / final.
        ini = "Ø"
        fin = base
        for cand in initials:
            if base.startswith(cand) and base != cand:
                ini = cand
                fin = base[len(cand) :]
                break
            if base == cand:
                # Rare but handle it; treat as (cand + Ø) to avoid empty final.
                ini = cand
                fin = "Ø"
                break

        if ini not in phoneme_id_map:
            raise RuntimeError(f"pinyin 声母不在 phoneme_id_map 中：{ini} (from {py})")
        if fin not in phoneme_id_map:
            raise RuntimeError(f"pinyin 韵母不在 phoneme_id_map 中：{fin} (from {py})")
        if tone not in phoneme_id_map:
            raise RuntimeError(f"pinyin 声调不在 phoneme_id_map 中：{tone} (from {py})")

        out.extend([ini, fin, tone])

    return out


def _tokens_to_ids(tokens: List[str], *, phoneme_id_map: Dict[str, List[int]]) -> List[int]:
    ids: List[int] = []
    for t in tokens:
        v = phoneme_id_map.get(t)
        if not v:
            raise RuntimeError(f"token 不在 phoneme_id_map 中：{t!r}")
        for x in v:
            ids.append(_safe_int(x, 0))
    return ids


@dataclass
class PiperOnnxPinyinTTS:
    model_path: Path
    config_path: Path
    config: dict
    provider: str = "cpu"  # cpu | cuda
    num_threads: int = 2

    _sess: object = field(default=None, init=False, repr=False)
    _phoneme_id_map: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _sample_rate: int = field(default=22050, init=False, repr=False)

    def _ensure_session(self) -> None:
        if self._sess is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise RuntimeError("缺少依赖 onnxruntime，无法运行 piper onnx。请先 pip install onnxruntime") from exc

        audio_cfg = self.config.get("audio") or {}
        self._sample_rate = _safe_int(audio_cfg.get("sample_rate"), 22050)
        self._phoneme_id_map = dict(self.config.get("phoneme_id_map") or {})
        if not self._phoneme_id_map:
            raise RuntimeError(f"无效的 Piper config：phoneme_id_map 为空 ({self.config_path})")

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(self.num_threads))
        so.inter_op_num_threads = 1
        try:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass

        providers: List[str] = ["CPUExecutionProvider"]
        if (self.provider or "").strip().lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._sess = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)

    def synthesize_to_wav(self, text: str, wav_path: str, *, speed: float = 1.0) -> str:
        self._ensure_session()

        text = (text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        tokens = _pinyin_text_to_tokens(text, phoneme_id_map=self._phoneme_id_map)
        if not tokens:
            raise RuntimeError("无法将文本转换为 pinyin tokens（可能包含大量非中文字符）。")

        # Add BOS/EOS if present in vocab.
        bos = "^" if "^" in self._phoneme_id_map else None
        eos = "$" if "$" in self._phoneme_id_map else None
        if bos:
            tokens = [bos] + tokens
        if eos:
            tokens = tokens + [eos]

        ids = _tokens_to_ids(tokens, phoneme_id_map=self._phoneme_id_map)

        noise_scale, length_scale, noise_w = _get_inference_scales(self.config, speed=speed)

        import numpy as np  # type: ignore

        x = np.asarray(ids, dtype=np.int64)[None, :]
        x_lens = np.asarray([x.shape[1]], dtype=np.int64)
        scales = np.asarray([noise_scale, length_scale, noise_w], dtype=np.float32)

        feed = {
            "input": x,
            "input_lengths": x_lens,
            "scales": scales,
        }

        outs = self._sess.run(None, feed)  # type: ignore[union-attr]
        y = outs[0]
        # y shape: (1, 1, 1, T)
        audio = y.reshape(-1).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16
            w.setframerate(int(self._sample_rate))
            w.writeframes(pcm.tobytes())

        return wav_path


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


def create_tts(*, speaker: Optional[int] = None):
    """Create a Piper TTS wrapper.

    This backend is intended for voices that are NOT compatible with sherpa-onnx's
    piper phonemizer (e.g. phoneme_type=pinyin voices like xiao_ya/chaowen).
    """
    model_dir = _resolve_model_dir()
    model_path, config_path = _resolve_voice_and_config(model_dir)

    cfg = _load_config_or_none(config_path)
    phoneme_type = ((cfg or {}).get("phoneme_type") or "").strip().lower()
    pid_map = (cfg or {}).get("phoneme_id_map") or {}
    has_multi_key = False
    if isinstance(pid_map, dict):
        has_multi_key = any(isinstance(k, str) and not _is_single_codepoint(k) for k in pid_map.keys())

    # Piper's official runtime expects phoneme_id_map keys to be single unicode codepoints.
    # The zh pinyin voices (xiao_ya/chaowen) use multi-character tokens, so we run them
    # in-process with onnxruntime + pypinyin instead of calling piper.exe.
    if phoneme_type == "pinyin" and has_multi_key and config_path is not None:
        provider = os.environ.get("PIPER_NATIVE_PROVIDER", "cpu").strip().lower() or "cpu"
        num_threads = _safe_int(os.environ.get("PIPER_NATIVE_NUM_THREADS", "").strip(), 2)
        return PiperOnnxPinyinTTS(
            model_path=model_path,
            config_path=config_path,
            config=cfg or {},
            provider=provider,
            num_threads=num_threads,
        )

    # Fallback: call the official piper executable (for configs that it supports).
    piper_bin = _resolve_piper_bin()
    if speaker is None:
        v = os.environ.get("PIPER_SPEAKER", "").strip()
        if v:
            try:
                speaker = int(v)
            except ValueError:
                speaker = None

    if (not piper_bin.exists()) or (not piper_bin.is_file()):
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
    tts,
    text: str,
    wav_path: str,
    *,
    speed: float = 1.0,
) -> tuple[str, float]:
    wav_path = tts.synthesize_to_wav(text, wav_path, speed=speed)
    return wav_path, _wav_duration_sec(wav_path)

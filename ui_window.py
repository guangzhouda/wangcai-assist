import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent


def _safe_get_input_devices() -> list[str]:
    try:
        from pvrecorder import PvRecorder  # type: ignore

        return list(PvRecorder.get_available_devices())
    except Exception:
        return []


def _safe_get_output_devices() -> list[str]:
    try:
        import sounddevice as sd  # type: ignore

        devs = sd.query_devices()
        out: list[str] = []
        for i, d in enumerate(devs):
            try:
                if int(d.get("max_output_channels", 0)) > 0:
                    out.append(f"[{i}] {d.get('name', '')}")
            except Exception:
                pass
        return out
    except Exception:
        return []


def _parse_choice_to_index(s: str) -> int:
    s = (s or "").strip()
    if s.startswith("Default"):
        return -1
    if s.startswith("[") and "]" in s:
        try:
            return int(s[1 : s.index("]")])
        except Exception:
            return -1
    return -1


def _create_process(
    *,
    python_exe: str,
    picovoice_key: str,
    deepseek_key: str,
    mic_device_index: int,
    kws_device_index: int,
    output_device_index: int,
    kws_sensitivity: float,
    asr_provider: str,
    tts_engine: str,
    cosyvoice_prompt_wav: str = "",
    cosyvoice_prompt_text: str = "",
    openvoice_ckpt_dir: str = "",
    openvoice_ref_wav: str = "",
    openvoice_device: str = "",
    openvoice_base_engine: str = "",
    openvoice_piper_provider: str = "",
) -> subprocess.Popen:
    env = os.environ.copy()
    if picovoice_key.strip():
        env["PICOVOICE_ACCESS_KEY"] = picovoice_key.strip()
    if deepseek_key.strip():
        env["DEEPSEEK_API_KEY"] = deepseek_key.strip()

    env["MIC_DEVICE_INDEX"] = str(mic_device_index)
    env["KWS_DEVICE_INDEX"] = str(kws_device_index)
    if output_device_index >= 0:
        # Only set when user explicitly chooses a speaker. If it's -1, keep system default.
        env["OUTPUT_DEVICE_INDEX"] = str(output_device_index)
    else:
        env.pop("OUTPUT_DEVICE_INDEX", None)

    env["KWS_SENSITIVITY"] = str(kws_sensitivity)
    env["ASR_PROVIDER"] = (asr_provider or "cuda").strip()
    env["TTS_ENGINE"] = (tts_engine or "piper").strip()

    if cosyvoice_prompt_wav.strip():
        env["COSYVOICE_PROMPT_WAV"] = cosyvoice_prompt_wav.strip()
    if cosyvoice_prompt_text.strip():
        env["COSYVOICE_PROMPT_TEXT"] = cosyvoice_prompt_text.strip()

    if openvoice_ckpt_dir.strip():
        env["OPENVOICE_CKPT_DIR"] = openvoice_ckpt_dir.strip()
    if openvoice_ref_wav.strip():
        env["OPENVOICE_REF_WAV"] = openvoice_ref_wav.strip()
    if openvoice_device.strip():
        env["OPENVOICE_DEVICE"] = openvoice_device.strip()
    if openvoice_base_engine.strip():
        env["OPENVOICE_BASE_ENGINE"] = openvoice_base_engine.strip()
    if openvoice_piper_provider.strip():
        env["OPENVOICE_PIPER_PROVIDER"] = openvoice_piper_provider.strip()

    cmd = [python_exe, str(PROJECT_DIR / "main.py")]

    creationflags = 0
    if os.name == "nt":
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )


def _terminate_process(proc: subprocess.Popen, *, timeout_sec: float = 3.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except Exception:
            pass

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        proc.terminate()
    except Exception:
        pass

    deadline = time.time() + 1.5
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        proc.kill()
    except Exception:
        pass


@dataclass
class UiState:
    proc: Optional[subprocess.Popen] = None


def main() -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise SystemExit("缺少 tkinter（你的 Python 可能没带 Tk 支持）。") from exc

    state = UiState()

    root = tk.Tk()
    root.title("Wangcai Assist (Window)")
    root.geometry("820x520")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # --- Top controls ---
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    # Vars
    status_var = tk.StringVar(value="Idle")
    picovoice_var = tk.StringVar(value=os.environ.get("PICOVOICE_ACCESS_KEY", ""))
    deepseek_var = tk.StringVar(value=os.environ.get("DEEPSEEK_API_KEY", ""))

    input_devices = _safe_get_input_devices()
    input_choices = ["Default (-1)"] + [f"[{i}] {name}" for i, name in enumerate(input_devices)]

    output_choices = ["Default (-1)"]
    out_extra = _safe_get_output_devices()
    if out_extra:
        output_choices += out_extra
    else:
        output_choices[0] = "Default (-1) (install sounddevice to select speaker)"

    mic_var = tk.StringVar(value="Default (-1)")
    kws_var = tk.StringVar(value="Default (-1)")
    spk_var = tk.StringVar(value=output_choices[0])
    kws_sens_var = tk.DoubleVar(value=float(os.environ.get("KWS_SENSITIVITY", "0.5") or "0.5"))
    asr_provider_var = tk.StringVar(value=os.environ.get("ASR_PROVIDER", "cuda"))
    tts_engine_var = tk.StringVar(value=os.environ.get("TTS_ENGINE", "piper"))

    openvoice_ckpt_var = tk.StringVar(
        value=os.environ.get(
            "OPENVOICE_CKPT_DIR",
            str(PROJECT_DIR / "model" / "openvoice_v2" / "checkpoints_v2"),
        )
    )
    openvoice_ref_var = tk.StringVar(value=os.environ.get("OPENVOICE_REF_WAV", ""))
    openvoice_device_var = tk.StringVar(value=os.environ.get("OPENVOICE_DEVICE", "auto"))
    openvoice_base_var = tk.StringVar(value=os.environ.get("OPENVOICE_BASE_ENGINE", "piper"))
    openvoice_piper_provider_var = tk.StringVar(value=os.environ.get("OPENVOICE_PIPER_PROVIDER", "cpu"))

    # CosyVoice (optional)
    default_ref_wav = str(PROJECT_DIR / "myvoice.wav") if (PROJECT_DIR / "myvoice.wav").exists() else ""
    cosy_prompt_wav_var = tk.StringVar(value=os.environ.get("COSYVOICE_PROMPT_WAV", default_ref_wav))
    default_prompt_text = os.environ.get("COSYVOICE_PROMPT_TEXT", "")
    if not default_prompt_text:
        ref_txt = PROJECT_DIR / "myvoice-ref.txt"
        if ref_txt.exists():
            try:
                default_prompt_text = ref_txt.read_text(encoding="utf-8").strip()
            except Exception:
                default_prompt_text = ""
    if not default_prompt_text:
        default_prompt_text = "你好，我是旺财。"
    cosy_prompt_text_var = tk.StringVar(value=default_prompt_text)

    def add_row(r: int, label: str, widget) -> None:
        ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=r, column=1, sticky="ew", pady=4)

    top.columnconfigure(1, weight=1)

    add_row(0, "PICOVOICE_KEY", ttk.Entry(top, textvariable=picovoice_var, show="*"))
    add_row(1, "DEEPSEEK_KEY", ttk.Entry(top, textvariable=deepseek_var, show="*"))

    add_row(2, "麦克风 (ASR)", ttk.Combobox(top, textvariable=mic_var, values=input_choices, state="readonly"))
    add_row(3, "麦克风 (KWS)", ttk.Combobox(top, textvariable=kws_var, values=input_choices, state="readonly"))
    add_row(4, "扬声器 (输出)", ttk.Combobox(top, textvariable=spk_var, values=output_choices, state="readonly"))

    add_row(5, "KWS 灵敏度", ttk.Scale(top, variable=kws_sens_var, from_=0.1, to=0.9, orient="horizontal"))

    add_row(6, "ASR Provider", ttk.Combobox(top, textvariable=asr_provider_var, values=["cuda", "cpu"], state="readonly"))
    add_row(
        7,
        "TTS 引擎",
        ttk.Combobox(
            top,
            textvariable=tts_engine_var,
            values=["piper", "piper_native", "matcha", "cosyvoice", "openvoice"],
            state="readonly",
        ),
    )

    # OpenVoice extras
    ov = ttk.LabelFrame(top, text="OpenVoice (仅当 TTS_ENGINE=openvoice)", padding=8)
    ov.grid(row=0, column=2, rowspan=8, sticky="nsew", padx=(12, 0))
    ov.columnconfigure(1, weight=1)

    def ov_row(r: int, label: str, widget) -> None:
        ttk.Label(ov, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=3)
        widget.grid(row=r, column=1, sticky="ew", pady=3)

    ov_row(0, "CKPT_DIR", ttk.Entry(ov, textvariable=openvoice_ckpt_var))
    ov_row(1, "REF_WAV", ttk.Entry(ov, textvariable=openvoice_ref_var))
    ov_row(2, "DEVICE", ttk.Combobox(ov, textvariable=openvoice_device_var, values=["auto", "cuda", "cpu"], state="readonly"))
    ov_row(3, "BASE", ttk.Combobox(ov, textvariable=openvoice_base_var, values=["piper"], state="readonly"))
    ov_row(4, "PIPER_PROVIDER", ttk.Combobox(ov, textvariable=openvoice_piper_provider_var, values=["cpu", "cuda"], state="readonly"))

    cosy = ttk.LabelFrame(top, text="CosyVoice (仅当 TTS_ENGINE=cosyvoice)", padding=8)
    cosy.grid(row=8, column=2, sticky="nsew", padx=(12, 0), pady=(10, 0))
    cosy.columnconfigure(1, weight=1)

    ttk.Label(cosy, text="PROMPT_WAV").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
    ttk.Entry(cosy, textvariable=cosy_prompt_wav_var).grid(row=0, column=1, sticky="ew", pady=3)

    ttk.Label(cosy, text="PROMPT_TEXT").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
    ttk.Entry(cosy, textvariable=cosy_prompt_text_var).grid(row=1, column=1, sticky="ew", pady=3)

    # Buttons row
    btns = ttk.Frame(top)
    btns.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    btns.columnconfigure(2, weight=1)

    status_lbl = ttk.Label(btns, textvariable=status_var)
    status_lbl.grid(row=0, column=0, sticky="w")

    # --- Log area ---
    mid = ttk.Frame(root, padding=(10, 0, 10, 10))
    mid.pack(fill="both", expand=True)

    log_text = tk.Text(mid, height=20, wrap="word")
    log_text.pack(side="left", fill="both", expand=True)
    log_text.configure(state="disabled")

    scroll = ttk.Scrollbar(mid, command=log_text.yview)
    scroll.pack(side="right", fill="y")
    log_text.configure(yscrollcommand=scroll.set)

    def _append(line: str) -> None:
        log_text.configure(state="normal")
        log_text.insert("end", line)
        log_text.see("end")
        log_text.configure(state="disabled")

    def _set_status(v: str) -> None:
        status_var.set(v)

    def start_pipeline() -> None:
        if state.proc is not None and state.proc.poll() is None:
            return

        proc = _create_process(
            python_exe=sys.executable,
            picovoice_key=picovoice_var.get(),
            deepseek_key=deepseek_var.get(),
            mic_device_index=_parse_choice_to_index(mic_var.get()),
            kws_device_index=_parse_choice_to_index(kws_var.get()),
            output_device_index=_parse_choice_to_index(spk_var.get()),
            kws_sensitivity=float(kws_sens_var.get()),
            asr_provider=(asr_provider_var.get() or "cuda").strip(),
            tts_engine=(tts_engine_var.get() or "piper").strip(),
            cosyvoice_prompt_wav=cosy_prompt_wav_var.get(),
            cosyvoice_prompt_text=cosy_prompt_text_var.get(),
            openvoice_ckpt_dir=openvoice_ckpt_var.get(),
            openvoice_ref_wav=openvoice_ref_var.get(),
            openvoice_device=openvoice_device_var.get(),
            openvoice_base_engine=openvoice_base_var.get(),
            openvoice_piper_provider=openvoice_piper_provider_var.get(),
        )
        state.proc = proc
        _set_status("Running")
        _append("\n=== START ===\n")

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    root.after(0, _append, line)
            except Exception as exc:
                root.after(0, _append, f"\n[log reader error] {exc}\n")
            finally:
                code = proc.poll()
                root.after(0, _append, f"\n=== EXIT ({code}) ===\n")
                root.after(0, _set_status, "Idle")
                state.proc = None

        threading.Thread(target=reader, daemon=True).start()

    def stop_pipeline() -> None:
        proc = state.proc
        if proc is None or proc.poll() is not None:
            state.proc = None
            _set_status("Idle")
            return
        _append("\n=== STOP ===\n")
        _terminate_process(proc)
        state.proc = None
        _set_status("Idle")

    def clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    ttk.Button(btns, text="启动", command=start_pipeline).grid(row=0, column=1, padx=(10, 0))
    ttk.Button(btns, text="停止", command=stop_pipeline).grid(row=0, column=2, padx=(8, 0), sticky="w")
    ttk.Button(btns, text="清空日志", command=clear_log).grid(row=0, column=3, padx=(8, 0))

    def on_close() -> None:
        try:
            stop_pipeline()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple


PROJECT_DIR = Path(__file__).resolve().parent


def _safe_get_devices() -> list[str]:
    try:
        from pvrecorder import PvRecorder  # type: ignore

        return list(PvRecorder.get_available_devices())
    except Exception:
        return []


def _parse_device_choice(s: str) -> int:
    # Choices look like: "Default (-1)" or "[3] Microphone XYZ"
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
    kws_sensitivity: float,
    asr_provider: str,
    tts_engine: str,
) -> subprocess.Popen:
    env = os.environ.copy()
    if picovoice_key.strip():
        env["PICOVOICE_ACCESS_KEY"] = picovoice_key.strip()
    if deepseek_key.strip():
        env["DEEPSEEK_API_KEY"] = deepseek_key.strip()

    env["MIC_DEVICE_INDEX"] = str(mic_device_index)
    env["KWS_DEVICE_INDEX"] = str(kws_device_index)
    env["KWS_SENSITIVITY"] = str(kws_sensitivity)
    env["ASR_PROVIDER"] = asr_provider
    env["TTS_ENGINE"] = tts_engine

    # Run the integrated pipeline entry.
    cmd = [python_exe, str(PROJECT_DIR / "main.py")]

    creationflags = 0
    if os.name == "nt":
        # Create a new process group so we can terminate it more reliably.
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
    try:
        proc.terminate()
    except Exception:
        pass
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        proc.kill()
    except Exception:
        pass


def main() -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise SystemExit("缺少 tkinter（你的 Python 可能没带 Tk 支持）。") from exc

    devices = _safe_get_devices()
    device_choices = ["Default (-1)"] + [f"[{i}] {name}" for i, name in enumerate(devices)]

    root = tk.Tk()
    root.title("Wangcai Assist")
    root.geometry("420x320")
    root.minsize(420, 320)

    # State
    proc_ref: dict[str, Optional[subprocess.Popen]] = {"proc": None}

    status_var = tk.StringVar(value="Idle")
    topmost_var = tk.BooleanVar(value=True)
    picovoice_var = tk.StringVar(value=os.environ.get("PICOVOICE_ACCESS_KEY", ""))
    deepseek_var = tk.StringVar(value=os.environ.get("DEEPSEEK_API_KEY", ""))
    mic_var = tk.StringVar(value="Default (-1)")
    kws_var = tk.StringVar(value="Default (-1)")
    kws_sens_var = tk.DoubleVar(value=float(os.environ.get("KWS_SENSITIVITY", "0.5") or "0.5"))
    asr_provider_var = tk.StringVar(value=os.environ.get("ASR_PROVIDER", "cuda"))
    tts_engine_var = tk.StringVar(value=os.environ.get("TTS_ENGINE", "piper"))

    def set_status(s: str) -> None:
        status_var.set(s)

    def apply_topmost() -> None:
        try:
            root.attributes("-topmost", bool(topmost_var.get()))
        except Exception:
            pass

    def append_log(line: str) -> None:
        # Must be called on UI thread.
        log_text.configure(state="normal")
        log_text.insert("end", line)
        log_text.see("end")
        log_text.configure(state="disabled")

    def start_pipeline() -> None:
        if proc_ref["proc"] is not None and proc_ref["proc"].poll() is None:
            return

        python_exe = sys.executable
        proc = _create_process(
            python_exe=python_exe,
            picovoice_key=picovoice_var.get(),
            deepseek_key=deepseek_var.get(),
            mic_device_index=_parse_device_choice(mic_var.get()),
            kws_device_index=_parse_device_choice(kws_var.get()),
            kws_sensitivity=float(kws_sens_var.get()),
            asr_provider=(asr_provider_var.get() or "cuda").strip(),
            tts_engine=(tts_engine_var.get() or "piper").strip(),
        )
        proc_ref["proc"] = proc
        set_status("Running")
        append_log("\n=== START ===\n")

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    root.after(0, append_log, line)
            except Exception as exc:
                root.after(0, append_log, f"\n[log reader error] {exc}\n")
            finally:
                code = proc.poll()
                root.after(0, append_log, f"\n=== EXIT ({code}) ===\n")
                root.after(0, set_status, "Idle")
                proc_ref["proc"] = None

        threading.Thread(target=reader, daemon=True).start()

    def stop_pipeline() -> None:
        proc = proc_ref["proc"]
        if proc is None or proc.poll() is not None:
            set_status("Idle")
            proc_ref["proc"] = None
            return
        append_log("\n=== STOP ===\n")
        _terminate_process(proc)
        proc_ref["proc"] = None
        set_status("Idle")

    def on_close() -> None:
        stop_pipeline()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Layout
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    header = ttk.Frame(frm)
    header.pack(fill="x")

    ttk.Label(header, text="状态:").pack(side="left")
    ttk.Label(header, textvariable=status_var).pack(side="left", padx=(4, 0))

    ttk.Checkbutton(header, text="置顶", variable=topmost_var, command=apply_topmost).pack(side="right")

    ttk.Separator(frm).pack(fill="x", pady=8)

    grid = ttk.Frame(frm)
    grid.pack(fill="x")

    def add_row(row: int, label: str, widget) -> None:
        ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        widget.grid(row=row, column=1, sticky="ew", pady=3)

    grid.columnconfigure(1, weight=1)

    pic_entry = ttk.Entry(grid, textvariable=picovoice_var, show="*")
    add_row(0, "PICOVOICE_KEY", pic_entry)

    ds_entry = ttk.Entry(grid, textvariable=deepseek_var, show="*")
    add_row(1, "DEEPSEEK_KEY", ds_entry)

    mic_combo = ttk.Combobox(grid, textvariable=mic_var, values=device_choices, state="readonly")
    add_row(2, "麦克风 (ASR)", mic_combo)

    kws_combo = ttk.Combobox(grid, textvariable=kws_var, values=device_choices, state="readonly")
    add_row(3, "麦克风 (KWS)", kws_combo)

    sens = ttk.Scale(grid, variable=kws_sens_var, from_=0.1, to=0.9, orient="horizontal")
    add_row(4, "KWS 灵敏度", sens)

    asr_combo = ttk.Combobox(grid, textvariable=asr_provider_var, values=["cuda", "cpu"], state="readonly")
    add_row(5, "ASR Provider", asr_combo)

    tts_combo = ttk.Combobox(grid, textvariable=tts_engine_var, values=["piper", "cosyvoice"], state="readonly")
    add_row(6, "TTS 引擎", tts_combo)

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(10, 6))
    ttk.Button(btns, text="启动", command=start_pipeline).pack(side="left")
    ttk.Button(btns, text="停止", command=stop_pipeline).pack(side="left", padx=(8, 0))

    ttk.Label(
        frm,
        text="提示：启动后说“旺财”唤醒；对话中说“休眠/退出/再见”回到待机。",
        foreground="#666666",
        wraplength=380,
        justify="left",
    ).pack(fill="x", pady=(0, 6))

    log_frame = ttk.Frame(frm)
    log_frame.pack(fill="both", expand=True)

    log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
    scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    apply_topmost()
    root.mainloop()


if __name__ == "__main__":
    main()


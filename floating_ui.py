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


def _safe_get_devices() -> list[str]:
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


def _parse_device_choice(s: str) -> int:
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
    env["OUTPUT_DEVICE_INDEX"] = str(output_device_index)
    env["KWS_SENSITIVITY"] = str(kws_sensitivity)
    env["ASR_PROVIDER"] = (asr_provider or "cuda").strip()
    env["TTS_ENGINE"] = (tts_engine or "piper").strip()

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
        # New process group lets us send CTRL_BREAK_EVENT for a graceful stop.
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

    # Try graceful stop first (Windows).
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
class DockConfig:
    side: str = "left"  # "left" | "right"
    handle_w: int = 16
    panel_w: int = 420
    height: int = 480
    margin_y: int = 80
    alpha: float = 0.98


def main() -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise SystemExit("缺少 tkinter（你的 Python 可能没带 Tk 支持）。") from exc

    cfg = DockConfig(side=os.environ.get("UI_DOCK_SIDE", "left").strip().lower() or "left")
    if cfg.side not in ("left", "right"):
        cfg.side = "left"

    devices = _safe_get_devices()
    device_choices = ["Default (-1)"] + [f"[{i}] {name}" for i, name in enumerate(devices)]

    out_devs = _safe_get_output_devices()
    out_choices = ["Default (-1)"] + out_devs

    root = tk.Tk()
    root.title("Wangcai Assist")

    # Borderless + always on top + slightly transparent.
    root.overrideredirect(True)
    try:
        root.attributes("-topmost", True)
        root.attributes("-alpha", cfg.alpha)
        if os.name == "nt":
            root.attributes("-toolwindow", True)
    except Exception:
        pass

    # Theme (simple light look).
    BG = "#f8fafc"
    BORDER = "#cbd5e1"
    TEXT = "#0f172a"
    MUTED = "#475569"
    ACCENT = "#0ea5e9"
    RUNNING = "#22c55e"
    IDLE = "#94a3b8"

    root.configure(bg=BG)

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("W.TFrame", background=BG)
    style.configure("W.TLabel", background=BG, foreground=TEXT)
    style.configure("W.Muted.TLabel", background=BG, foreground=MUTED)
    style.configure("W.TButton", padding=(10, 6))
    style.configure("W.TEntry", padding=(6, 4))
    style.configure("W.TCombobox", padding=(6, 4))

    # State
    proc_ref: dict[str, Optional[subprocess.Popen]] = {"proc": None}
    tray_ref: dict[str, object] = {"icon": None}

    window_visible = True
    expanded = False
    dragging = False
    drag_start: Optional[tuple[int, int]] = None  # (mouse_y, win_y)

    status_var = tk.StringVar(value="Idle")
    picovoice_var = tk.StringVar(value=os.environ.get("PICOVOICE_ACCESS_KEY", ""))
    deepseek_var = tk.StringVar(value=os.environ.get("DEEPSEEK_API_KEY", ""))
    mic_var = tk.StringVar(value="Default (-1)")
    kws_var = tk.StringVar(value="Default (-1)")
    speaker_var = tk.StringVar(value="Default (-1)")
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

    def _screen_w() -> int:
        return int(root.winfo_screenwidth())

    def _screen_h() -> int:
        return int(root.winfo_screenheight())

    # Initial position: about 1/3 from top.
    win_y = min(max(cfg.margin_y, int(_screen_h() * 0.22)), _screen_h() - cfg.height - cfg.margin_y)

    def _place_window() -> None:
        nonlocal win_y
        w = cfg.panel_w if expanded else cfg.handle_w
        h = cfg.height

        win_y = min(max(0, win_y), max(0, _screen_h() - h))

        if cfg.side == "left":
            x = 0
        else:
            if expanded:
                x = _screen_w() - cfg.panel_w
            else:
                x = _screen_w() - cfg.handle_w

        root.geometry(f"{w}x{h}+{x}+{win_y}")

    def _set_visible(v: bool) -> None:
        nonlocal window_visible
        window_visible = v
        if v:
            root.deiconify()
            _place_window()
        else:
            root.withdraw()

    def _update_handle() -> None:
        handle_canvas.delete("all")
        st = status_var.get()
        dot = RUNNING if st == "Running" else IDLE

        handle_canvas.configure(bg=ACCENT)
        handle_canvas.create_rectangle(0, 0, cfg.handle_w, cfg.height, fill=ACCENT, outline=ACCENT)
        handle_canvas.create_oval(4, 6, 12, 14, fill=dot, outline="")

        if cfg.side == "left":
            ch = "<" if expanded else ">"
        else:
            ch = ">" if expanded else "<"
        handle_canvas.create_text(
            cfg.handle_w // 2,
            cfg.height // 2,
            text=ch,
            fill="white",
            font=("Segoe UI", 14, "bold"),
        )

    def _repack() -> None:
        # Pack order depends on dock side (handle stays at screen edge).
        for w in (panel_frame, handle_frame):
            w.pack_forget()

        if cfg.side == "left":
            handle_frame.pack(side="left", fill="y")
            if expanded:
                panel_frame.pack(side="left", fill="both", expand=True)
        else:
            if expanded:
                panel_frame.pack(side="left", fill="both", expand=True)
            handle_frame.pack(side="right", fill="y")

        _place_window()
        _update_handle()

    def _set_expanded(v: bool) -> None:
        nonlocal expanded
        expanded = v
        _repack()
        if expanded:
            try:
                root.focus_force()
            except Exception:
                pass

    def toggle_panel() -> None:
        if not window_visible:
            _set_visible(True)
            _set_expanded(False)
            return
        _set_expanded(not expanded)

    def _append_log(line: str) -> None:
        log_text.configure(state="normal")
        log_text.insert("end", line)
        log_text.see("end")
        log_text.configure(state="disabled")

    def _set_status(s: str) -> None:
        status_var.set(s)
        _update_handle()

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
            output_device_index=_parse_device_choice(speaker_var.get()),
            kws_sensitivity=float(kws_sens_var.get()),
            asr_provider=(asr_provider_var.get() or "cuda").strip(),
            tts_engine=(tts_engine_var.get() or "piper").strip(),
            openvoice_ckpt_dir=openvoice_ckpt_var.get(),
            openvoice_ref_wav=openvoice_ref_var.get(),
            openvoice_device=openvoice_device_var.get(),
            openvoice_base_engine=openvoice_base_var.get(),
            openvoice_piper_provider=openvoice_piper_provider_var.get(),
        )
        proc_ref["proc"] = proc
        _set_status("Running")
        _append_log("\n=== START ===\n")

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    root.after(0, _append_log, line)
            except Exception as exc:
                root.after(0, _append_log, f"\n[log reader error] {exc}\n")
            finally:
                code = proc.poll()
                root.after(0, _append_log, f"\n=== EXIT ({code}) ===\n")
                root.after(0, _set_status, "Idle")
                proc_ref["proc"] = None

        threading.Thread(target=reader, daemon=True).start()

    def stop_pipeline() -> None:
        proc = proc_ref["proc"]
        if proc is None or proc.poll() is not None:
            proc_ref["proc"] = None
            _set_status("Idle")
            return
        _append_log("\n=== STOP ===\n")
        _terminate_process(proc)
        proc_ref["proc"] = None
        _set_status("Idle")

    def _quit_app() -> None:
        stop_pipeline()
        icon = tray_ref.get("icon")
        try:
            if icon is not None:
                icon.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        root.destroy()

    def _toggle_visible() -> None:
        _set_visible(not window_visible)

    # Root container with 1px border (borderless windows look odd otherwise).
    outer = tk.Frame(root, bg=BG, highlightthickness=1, highlightbackground=BORDER)
    outer.pack(fill="both", expand=True)

    # Handle (always visible).
    handle_frame = tk.Frame(outer, width=cfg.handle_w, bg=ACCENT)
    handle_frame.pack_propagate(False)
    handle_canvas = tk.Canvas(
        handle_frame,
        width=cfg.handle_w,
        height=cfg.height,
        bg=ACCENT,
        highlightthickness=0,
    )
    handle_canvas.pack(fill="both", expand=True)

    # Panel (only packed when expanded).
    panel_frame = ttk.Frame(outer, style="W.TFrame", padding=10)

    # Panel header
    header = ttk.Frame(panel_frame, style="W.TFrame")
    header.pack(fill="x")

    title = ttk.Label(header, text="旺财助手", style="W.TLabel", font=("Segoe UI", 11, "bold"))
    title.pack(side="left")

    status_chip = ttk.Label(header, textvariable=status_var, style="W.Muted.TLabel")
    status_chip.pack(side="left", padx=(8, 0))

    btn_hide = ttk.Button(header, text="隐藏", style="W.TButton", command=_toggle_visible)
    btn_hide.pack(side="right")

    btn_quit = ttk.Button(header, text="退出", style="W.TButton", command=_quit_app)
    btn_quit.pack(side="right", padx=(6, 6))

    # Controls
    controls = ttk.Frame(panel_frame, style="W.TFrame")
    controls.pack(fill="x", pady=(10, 8))
    controls.columnconfigure(1, weight=1)

    def add_row(row: int, label: str, widget) -> None:
        ttk.Label(controls, text=label, style="W.Muted.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    pic_entry = ttk.Entry(controls, textvariable=picovoice_var, show="*", style="W.TEntry")
    add_row(0, "PICOVOICE_KEY", pic_entry)

    ds_entry = ttk.Entry(controls, textvariable=deepseek_var, show="*", style="W.TEntry")
    add_row(1, "DEEPSEEK_KEY", ds_entry)

    mic_combo = ttk.Combobox(controls, textvariable=mic_var, values=device_choices, state="readonly", style="W.TCombobox")
    add_row(2, "麦克风 (ASR)", mic_combo)

    kws_combo = ttk.Combobox(controls, textvariable=kws_var, values=device_choices, state="readonly", style="W.TCombobox")
    add_row(3, "麦克风 (KWS)", kws_combo)

    spk_combo = ttk.Combobox(controls, textvariable=speaker_var, values=out_choices, state="readonly", style="W.TCombobox")
    add_row(4, "扬声器 (输出)", spk_combo)

    sens = ttk.Scale(controls, variable=kws_sens_var, from_=0.1, to=0.9, orient="horizontal")
    add_row(5, "KWS 灵敏度", sens)

    asr_combo = ttk.Combobox(controls, textvariable=asr_provider_var, values=["cuda", "cpu"], state="readonly", style="W.TCombobox")
    add_row(6, "ASR Provider", asr_combo)

    tts_combo = ttk.Combobox(
        controls,
        textvariable=tts_engine_var,
        values=["piper", "piper_native", "matcha", "cosyvoice", "openvoice"],
        state="readonly",
        style="W.TCombobox",
    )
    add_row(7, "TTS 引擎", tts_combo)

    # OpenVoice V2 (shown only when TTS_ENGINE=openvoice)
    ov_ckpt_label = ttk.Label(controls, text="OV CKPT_DIR", style="W.Muted.TLabel")
    ov_ckpt_entry = ttk.Entry(controls, textvariable=openvoice_ckpt_var, style="W.TEntry")
    ov_ckpt_label.grid(row=8, column=0, sticky="w", padx=(0, 8), pady=4)
    ov_ckpt_entry.grid(row=8, column=1, sticky="ew", pady=4)

    ov_ref_label = ttk.Label(controls, text="OV REF_WAV", style="W.Muted.TLabel")
    ov_ref_entry = ttk.Entry(controls, textvariable=openvoice_ref_var, style="W.TEntry")
    ov_ref_label.grid(row=9, column=0, sticky="w", padx=(0, 8), pady=4)
    ov_ref_entry.grid(row=9, column=1, sticky="ew", pady=4)

    ov_dev_label = ttk.Label(controls, text="OV DEVICE", style="W.Muted.TLabel")
    ov_dev_combo = ttk.Combobox(
        controls,
        textvariable=openvoice_device_var,
        values=["auto", "cuda", "cpu"],
        state="readonly",
        style="W.TCombobox",
    )
    ov_dev_label.grid(row=10, column=0, sticky="w", padx=(0, 8), pady=4)
    ov_dev_combo.grid(row=10, column=1, sticky="ew", pady=4)

    ov_base_label = ttk.Label(controls, text="OV BASE", style="W.Muted.TLabel")
    ov_base_combo = ttk.Combobox(
        controls,
        textvariable=openvoice_base_var,
        values=["piper"],
        state="readonly",
        style="W.TCombobox",
    )
    ov_base_label.grid(row=11, column=0, sticky="w", padx=(0, 8), pady=4)
    ov_base_combo.grid(row=11, column=1, sticky="ew", pady=4)

    ov_piper_label = ttk.Label(controls, text="OV Piper", style="W.Muted.TLabel")
    ov_piper_combo = ttk.Combobox(
        controls,
        textvariable=openvoice_piper_provider_var,
        values=["cpu", "cuda"],
        state="readonly",
        style="W.TCombobox",
    )
    ov_piper_label.grid(row=12, column=0, sticky="w", padx=(0, 8), pady=4)
    ov_piper_combo.grid(row=12, column=1, sticky="ew", pady=4)

    openvoice_widgets = [
        ov_ckpt_label,
        ov_ckpt_entry,
        ov_ref_label,
        ov_ref_entry,
        ov_dev_label,
        ov_dev_combo,
        ov_base_label,
        ov_base_combo,
        ov_piper_label,
        ov_piper_combo,
    ]

    def update_openvoice_visibility(*_args) -> None:
        enabled = (tts_engine_var.get() or "").strip().lower() == "openvoice"
        for w in openvoice_widgets:
            if enabled:
                w.grid()
            else:
                w.grid_remove()

    tts_engine_var.trace_add("write", update_openvoice_visibility)
    update_openvoice_visibility()

    btns = ttk.Frame(panel_frame, style="W.TFrame")
    btns.pack(fill="x", pady=(0, 8))

    ttk.Button(btns, text="启动", style="W.TButton", command=start_pipeline).pack(side="left")
    ttk.Button(btns, text="停止", style="W.TButton", command=stop_pipeline).pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="收起", style="W.TButton", command=lambda: _set_expanded(False)).pack(side="right")

    tip = ttk.Label(
        panel_frame,
        text="提示：启动后说“旺财”唤醒；对话中说“休眠/退出/再见”回到待机。",
        style="W.Muted.TLabel",
        wraplength=cfg.panel_w - 40,
        justify="left",
    )
    tip.pack(fill="x", pady=(0, 8))

    # Logs (small, scrollable)
    log_frame = ttk.Frame(panel_frame, style="W.TFrame")
    log_frame.pack(fill="both", expand=True)

    log_text = tk.Text(
        log_frame,
        height=8,
        wrap="word",
        state="disabled",
        bg="white",
        fg=TEXT,
        insertbackground=TEXT,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    # Click-to-expand handle + drag to move vertically
    def on_handle_press(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal drag_start, dragging
        dragging = False
        drag_start = (int(event.y_root), int(win_y))

    def on_handle_drag(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal win_y, dragging
        if drag_start is None:
            return
        y0, wy0 = drag_start
        dy = int(event.y_root) - y0
        if abs(dy) > 3:
            dragging = True
        win_y = wy0 + dy
        _place_window()

    def on_handle_release(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal drag_start
        drag_start = None
        if not dragging:
            toggle_panel()

    for w in (handle_frame, handle_canvas):
        w.bind("<ButtonPress-1>", on_handle_press)
        w.bind("<B1-Motion>", on_handle_drag)
        w.bind("<ButtonRelease-1>", on_handle_release)

    # Collapse when focus is lost (click elsewhere) to behave like "auto-hide".
    def on_focus_out(event) -> None:  # type: ignore[no-untyped-def]
        if expanded:
            _set_expanded(False)

    root.bind("<FocusOut>", on_focus_out)

    # Key shortcuts
    root.bind("<Escape>", lambda e: _set_expanded(False))
    root.bind("<Control-q>", lambda e: _quit_app())

    # Dock side switch via hotkey (Ctrl+Shift+D)
    def _toggle_side() -> None:
        nonlocal cfg
        cfg.side = "right" if cfg.side == "left" else "left"
        _repack()

    root.bind("<Control-Shift-d>", lambda e: _toggle_side())

    # Tray icon (optional dependency)
    def _start_tray() -> None:
        try:
            import pystray  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except Exception:
            root.after(
                0,
                _append_log,
                "\n[tray] 未安装 pystray，无法启用托盘图标。可执行：pip install pystray\n",
            )
            return

        def make_image() -> "Image.Image":  # type: ignore[name-defined]
            img = Image.new("RGB", (64, 64), (14, 165, 233))
            d = ImageDraw.Draw(img)
            d.ellipse((10, 10, 54, 54), fill=(248, 250, 252))
            d.text((26, 18), "W", fill=(15, 23, 42))
            return img

        def set_side(side: str) -> None:
            nonlocal cfg
            side = (side or "").strip().lower()
            if side not in ("left", "right"):
                return
            cfg.side = side
            _repack()

        def ui_call(fn):
            def _inner(icon, item):  # noqa: ANN001
                root.after(0, fn)

            return _inner

        menu = pystray.Menu(
            pystray.MenuItem("展开/收起", ui_call(toggle_panel)),
            pystray.MenuItem("显示/隐藏", ui_call(_toggle_visible)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "靠左",
                ui_call(lambda: set_side("left")),
                checked=lambda item: cfg.side == "left",  # noqa: ARG005
            ),
            pystray.MenuItem(
                "靠右",
                ui_call(lambda: set_side("right")),
                checked=lambda item: cfg.side == "right",  # noqa: ARG005
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("启动", ui_call(start_pipeline)),
            pystray.MenuItem("停止", ui_call(stop_pipeline)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", ui_call(_quit_app)),
        )

        icon = pystray.Icon("wangcai-assist", make_image(), "Wangcai Assist", menu)
        tray_ref["icon"] = icon
        icon.run()

    threading.Thread(target=_start_tray, daemon=True).start()

    # Initial state: collapsed handle on the edge.
    _set_expanded(False)
    _place_window()
    _update_handle()

    # Ensure first show
    _set_visible(True)
    root.mainloop()


if __name__ == "__main__":
    main()

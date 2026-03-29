"""
control_panel.py — Session Control Panel
-----------------------------------------
A small floating window that lets the user control the
captioning session without touching the terminal.

Controls:
    - Start / Stop transcription
    - Clear captions
    - Language selector (auto-detect or specific language)
    - Toggle caption overlay visibility
    - Live session stats (duration, caption count, detected language)
"""

import tkinter as tk
from tkinter import ttk
import time
import threading


# Languages Whisper supports — displayed as (Label, whisper_code)
LANGUAGES = [
    ("Auto Detect",  None),
    ("English",      "en"),
    ("Spanish",      "es"),
    ("French",       "fr"),
    ("German",       "de"),
    ("Arabic",       "ar"),
    ("Chinese",      "zh"),
    ("Japanese",     "ja"),
    ("Korean",       "ko"),
    ("Portuguese",   "pt"),
    ("Russian",      "ru"),
    ("Hindi",        "hi"),
    ("Italian",      "it"),
]


class ControlPanel:
    """
    Floating control panel for the captioning session.
    Communicates with the transcription worker via shared
    state flags — no direct threading coupling needed.
    """

    BG_COLOR      = "#1E1E1E"
    ACCENT_COLOR  = "#0078D4"   # Windows blue
    TEXT_COLOR    = "#FFFFFF"
    SUBTLE_COLOR  = "#888888"
    SUCCESS_COLOR = "#4CAF50"
    DANGER_COLOR  = "#F44336"
    PANEL_WIDTH   = 320
    PANEL_HEIGHT  = 420

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Caption Control Panel")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Position top-right of screen
        screen_w = self.root.winfo_screenwidth()
        x = screen_w - self.PANEL_WIDTH - 20
        self.root.geometry(f"{self.PANEL_WIDTH}x{self.PANEL_HEIGHT}+{x}+20")

        # --- Shared state (read by whisper_worker) ---
        self.is_running       = False
        self.selected_language = None   # None = auto-detect
        self.caption_count    = 0
        self.detected_language = tk.StringVar(value="—")
        self._session_start   = None
        self._caption_display = None    # Reference to CaptionDisplay, set externally

        self._build_ui()

        # Start the clock updater
        self._update_clock()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 16, "pady": 6}

        # --- Header ---
        header = tk.Frame(self.root, bg=self.ACCENT_COLOR)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Live Caption Control",
            font=("Arial", 13, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.ACCENT_COLOR,
            pady=10,
        ).pack()

        # --- Status indicator ---
        status_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        status_frame.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(status_frame, text="Status", font=("Arial", 9),
                 fg=self.SUBTLE_COLOR, bg=self.BG_COLOR).pack(anchor="w")

        self.status_canvas = tk.Canvas(
            status_frame, width=12, height=12,
            bg=self.BG_COLOR, highlightthickness=0
        )
        self.status_canvas.pack(side="left", pady=2)
        self.status_dot = self.status_canvas.create_oval(2, 2, 10, 10, fill="#555555")

        self.status_var = tk.StringVar(value="Stopped")
        tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Arial", 11, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR,
        ).pack(side="left", padx=8)

        # --- Start / Stop button ---
        btn_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        btn_frame.pack(fill="x", padx=16, pady=6)

        self.toggle_btn = tk.Button(
            btn_frame,
            text="Start",
            command=self._toggle_transcription,
            font=("Arial", 11, "bold"),
            fg=self.TEXT_COLOR,
            bg=self.SUCCESS_COLOR,
            activebackground="#45A049",
            activeforeground=self.TEXT_COLOR,
            relief="flat",
            cursor="hand2",
            width=10,
            pady=6,
        )
        self.toggle_btn.pack(side="left")

        # --- Clear captions button ---
        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            command=self._clear_captions,
            font=("Arial", 11),
            fg=self.TEXT_COLOR,
            bg="#444444",
            activebackground="#555555",
            activeforeground=self.TEXT_COLOR,
            relief="flat",
            cursor="hand2",
            width=8,
            pady=6,
        )
        self.clear_btn.pack(side="left", padx=(10, 0))

        # --- Toggle overlay visibility ---
        self.overlay_visible = tk.BooleanVar(value=True)
        overlay_check = tk.Checkbutton(
            self.root,
            text="Show caption overlay",
            variable=self.overlay_visible,
            command=self._toggle_overlay,
            font=("Arial", 10),
            fg=self.TEXT_COLOR,
            bg=self.BG_COLOR,
            selectcolor=self.BG_COLOR,
            activebackground=self.BG_COLOR,
            activeforeground=self.TEXT_COLOR,
            cursor="hand2",
        )
        overlay_check.pack(anchor="w", **pad)

        # --- Divider ---
        tk.Frame(self.root, bg="#333333", height=1).pack(fill="x", padx=16, pady=4)

        # --- Language selector ---
        lang_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        lang_frame.pack(fill="x", padx=16, pady=6)

        tk.Label(lang_frame, text="Transcription Language",
                 font=("Arial", 9), fg=self.SUBTLE_COLOR,
                 bg=self.BG_COLOR).pack(anchor="w")

        self.lang_var = tk.StringVar(value="Auto Detect")
        lang_menu = ttk.Combobox(
            lang_frame,
            textvariable=self.lang_var,
            values=[label for label, _ in LANGUAGES],
            state="readonly",
            width=24,
            font=("Arial", 10),
        )
        lang_menu.pack(anchor="w", pady=4)
        lang_menu.bind("<<ComboboxSelected>>", self._on_language_change)

        # Style the combobox to match dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground="#2D2D2D",
            background="#2D2D2D",
            foreground=self.TEXT_COLOR,
            arrowcolor=self.TEXT_COLOR,
            bordercolor="#444444",
            lightcolor="#2D2D2D",
            darkcolor="#2D2D2D",
        )

        # --- Divider ---
        tk.Frame(self.root, bg="#333333", height=1).pack(fill="x", padx=16, pady=4)

        # --- Session stats ---
        stats_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        stats_frame.pack(fill="x", padx=16, pady=6)

        tk.Label(stats_frame, text="Session Stats",
                 font=("Arial", 9), fg=self.SUBTLE_COLOR,
                 bg=self.BG_COLOR).pack(anchor="w")

        stats_grid = tk.Frame(stats_frame, bg=self.BG_COLOR)
        stats_grid.pack(fill="x", pady=4)

        # Duration
        tk.Label(stats_grid, text="Duration", font=("Arial", 9),
                 fg=self.SUBTLE_COLOR, bg=self.BG_COLOR).grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.duration_var = tk.StringVar(value="00:00:00")
        tk.Label(stats_grid, textvariable=self.duration_var,
                 font=("Arial", 10, "bold"), fg=self.TEXT_COLOR,
                 bg=self.BG_COLOR).grid(row=0, column=1, sticky="w")

        # Caption count
        tk.Label(stats_grid, text="Captions", font=("Arial", 9),
                 fg=self.SUBTLE_COLOR, bg=self.BG_COLOR).grid(row=1, column=0, sticky="w", padx=(0, 20))
        self.count_var = tk.StringVar(value="0")
        tk.Label(stats_grid, textvariable=self.count_var,
                 font=("Arial", 10, "bold"), fg=self.TEXT_COLOR,
                 bg=self.BG_COLOR).grid(row=1, column=1, sticky="w")

        # Detected language
        tk.Label(stats_grid, text="Detected Lang", font=("Arial", 9),
                 fg=self.SUBTLE_COLOR, bg=self.BG_COLOR).grid(row=2, column=0, sticky="w", padx=(0, 20))
        tk.Label(stats_grid, textvariable=self.detected_language,
                 font=("Arial", 10, "bold"), fg=self.TEXT_COLOR,
                 bg=self.BG_COLOR).grid(row=2, column=1, sticky="w")

        # --- Footer ---
        tk.Label(
            self.root,
            text="Whisper ASR  |  Demo",
            font=("Arial", 8),
            fg="#444444",
            bg=self.BG_COLOR,
        ).pack(side="bottom", pady=8)

    # ------------------------------------------------------------------
    # Control actions
    # ------------------------------------------------------------------

    def _toggle_transcription(self):
        if not self.is_running:
            self._start()
        else:
            self._stop()

    def _start(self):
        self.is_running = True
        self._session_start = time.time()
        self.caption_count = 0
        self.count_var.set("0")
        self.duration_var.set("00:00:00")
        self.detected_language.set("—")

        self.toggle_btn.config(text="Stop", bg=self.DANGER_COLOR,
                               activebackground="#C62828")
        self.status_var.set("Running")
        self.status_canvas.itemconfig(self.status_dot, fill=self.SUCCESS_COLOR)

    def _stop(self):
        self.is_running = False

        self.toggle_btn.config(text="Start", bg=self.SUCCESS_COLOR,
                               activebackground="#45A049")
        self.status_var.set("Stopped")
        self.status_canvas.itemconfig(self.status_dot, fill="#555555")

        if self._caption_display:
            self._caption_display.clear()

    def _clear_captions(self):
        if self._caption_display:
            self._caption_display.clear()

    def _toggle_overlay(self):
        if self._caption_display:
            if self.overlay_visible.get():
                self._caption_display.root.deiconify()
            else:
                self._caption_display.root.withdraw()

    def _on_language_change(self, event=None):
        label = self.lang_var.get()
        # Look up the whisper code for the selected label
        for name, code in LANGUAGES:
            if name == label:
                self.selected_language = code
                break

    # ------------------------------------------------------------------
    # Public API — called from whisper_worker
    # ------------------------------------------------------------------

    def increment_caption_count(self, detected_lang: str = None):
        """Call this each time a valid caption is produced."""
        self.caption_count += 1
        self.count_var.set(str(self.caption_count))
        if detected_lang:
            self.detected_language.set(detected_lang.upper())

    def set_caption_display(self, caption_display):
        """Link the caption overlay so the panel can control it."""
        self._caption_display = caption_display

    def shutdown(self):
        self.is_running = False
        self.root.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

    # ------------------------------------------------------------------
    # Clock updater
    # ------------------------------------------------------------------

    def _update_clock(self):
        if self.is_running and self._session_start:
            elapsed = int(time.time() - self._session_start)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.duration_var.set(f"{h:02}:{m:02}:{s:02}")
        self.root.after(1000, self._update_clock)


# ------------------------------------------------------------------
# Standalone preview
# ------------------------------------------------------------------

if __name__ == "__main__":
    panel = ControlPanel()
    panel.run()
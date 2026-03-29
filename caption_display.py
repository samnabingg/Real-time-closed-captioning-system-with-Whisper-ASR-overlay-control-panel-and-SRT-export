"""
caption_display.py — Broadcast-style caption overlay
-----------------------------------------------------
Uses Toplevel so it shares the same Tkinter mainloop
as the ControlPanel root window.
"""

import tkinter as tk
import queue
import time


class CaptionDisplay:

    FONT_FAMILY  = "Arial"
    FONT_SIZE    = 28
    FONT_COLOR   = "white"
    BG_COLOR     = "black"
    MAX_LINES    = 2
    CLEAR_DELAY  = 5000   # ms before auto-fade after silence

    def __init__(self, master: tk.Tk):
        """
        Args:
            master: The root Tk() window (owned by ControlPanel).
        """
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title("Live Captions")

        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight()
        bar_height = 120

        self.window.geometry(f"{screen_w}x{bar_height}+0+{screen_h - bar_height - 60}")
        self.window.configure(bg=self.BG_COLOR)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.92)
        self.window.overrideredirect(True)

        # Prevent accidental close from destroying the master
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        # --- Caption label ---
        self.caption_var = tk.StringVar(value="")
        self.label = tk.Label(
            self.window,
            textvariable=self.caption_var,
            font=(self.FONT_FAMILY, self.FONT_SIZE, "bold"),
            fg=self.FONT_COLOR,
            bg=self.BG_COLOR,
            wraplength=screen_w - 40,
            justify="center",
            padx=20,
            pady=10,
        )
        self.label.pack(expand=True, fill="both")

        # --- Status bar ---
        self.status_var = tk.StringVar(value="")
        tk.Label(
            self.window,
            textvariable=self.status_var,
            font=(self.FONT_FAMILY, 10),
            fg="#888888",
            bg=self.BG_COLOR,
            anchor="e",
            padx=10,
        ).place(relx=1.0, rely=0.0, anchor="ne")

        # --- Internal state ---
        self.text_queue  = queue.Queue()
        self.lines       = []
        self.clear_timer = None
        self.running     = True

        self.window.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_text(self, text: str):
        self.text_queue.put(("text", text))

    def set_status(self, status: str):
        self.text_queue.put(("status", status))

    def clear(self):
        self.text_queue.put(("clear", ""))

    def hide(self):
        self.window.withdraw()

    def show(self):
        self.window.deiconify()

    def shutdown(self):
        self.running = False
        try:
            self.window.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                msg_type, content = self.text_queue.get_nowait()
                if msg_type == "text":
                    self._update_captions(content)
                elif msg_type == "status":
                    self.status_var.set(content)
                elif msg_type == "clear":
                    self._clear_captions()
        except queue.Empty:
            pass

        if self.running:
            self.window.after(50, self._poll_queue)

    def _update_captions(self, text: str):
        if self.clear_timer:
            self.window.after_cancel(self.clear_timer)

        words = text.split()
        current_line = ""
        new_lines = []
        for word in words:
            if len(current_line) + len(word) + 1 <= 60:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    new_lines.append(current_line)
                current_line = word
        if current_line:
            new_lines.append(current_line)

        self.lines.extend(new_lines)
        self.lines = self.lines[-self.MAX_LINES:]
        self.caption_var.set("\n".join(self.lines))
        self.label.config(fg=self.FONT_COLOR)

        self.clear_timer = self.window.after(self.CLEAR_DELAY, self._fade_out)

    def _fade_out(self):
        colors = ["white", "#CCCCCC", "#AAAAAA", "#888888",
                  "#666666", "#444444", "#222222", "#111111", "black"]
        self._animate_fade(colors, 0)

    def _animate_fade(self, colors, index):
        if index < len(colors):
            self.label.config(fg=colors[index])
            self.window.after(80, lambda: self._animate_fade(colors, index + 1))
        else:
            self._clear_captions()

    def _clear_captions(self):
        self.lines = []
        self.caption_var.set("")
        self.label.config(fg=self.FONT_COLOR)
        self.clear_timer = None
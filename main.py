"""
main.py — Real-Time Closed Captioning System
---------------------------------------------
Single Tk() root owned by ControlPanel.
CaptionDisplay is a Toplevel child of that root.
Whisper pipeline runs in a background daemon thread.

Usage:
    python main.py
"""

import threading
import asyncio
from audio_streaming import stream, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE
from caption_display import CaptionDisplay
from control_panel import ControlPanel
from whisper_worker import transcribe_worker


def run_transcription(caption_display: CaptionDisplay, control_panel: ControlPanel):
    """Whisper async pipeline — runs in a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        with stream:
            await transcribe_worker(
                caption_display=caption_display,
                control_panel=control_panel,
            )

    try:
        loop.run_until_complete(_run())
    except Exception as e:
        print(f"[Worker error]: {e}")
    finally:
        loop.close()


def main():
    print("=" * 55)
    print("  Real-Time Closed Captioning — Whisper ASR")
    print("=" * 55)
    print(f"  Device sample rate : {DEVICE_SAMPLE_RATE} Hz")
    print(f"  Whisper input rate : {WHISPER_SAMPLE_RATE} Hz")
    print(f"  Press Start on the control panel to begin.")
    print("=" * 55)

    # ControlPanel owns the single Tk() root
    control_panel = ControlPanel()

    # CaptionDisplay is a Toplevel child — shares the same mainloop
    caption_display = CaptionDisplay(master=control_panel.root)

    # Link them so the panel can show/hide the overlay
    control_panel.set_caption_display(caption_display)

    # Clean shutdown when control panel is closed
    def on_close():
        control_panel.is_running = False
        caption_display.shutdown()
        control_panel.root.destroy()

    control_panel.root.protocol("WM_DELETE_WINDOW", on_close)

    # Start Whisper pipeline in background daemon thread
    threading.Thread(
        target=run_transcription,
        args=(caption_display, control_panel),
        daemon=True,
        name="WhisperWorker",
    ).start()

    # Single mainloop — drives both windows
    control_panel.root.mainloop()

    print("\nSession ended.")


if __name__ == "__main__":
    main()
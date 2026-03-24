"""
srt_exporter.py — Real-Time SRT Subtitle File Writer
------------------------------------------------------
Receives transcription results with timestamps and writes
a valid .srt subtitle file that can be opened in VLC,
imported into Premiere Pro, YouTube, or any broadcast tool.

SRT format spec:
    [sequence number]
    [start] --> [end]
    [text]
    [blank line]
"""

import time
import threading
import os


class SRTExporter:
    """
    Thread-safe SRT subtitle writer.
    Call add_entry() from your transcription worker each time
    a new caption is ready. The file is written incrementally
    so it's always valid even if the session is interrupted.
    """

    def __init__(self, output_path: str = None):
        # Auto-generate filename with session timestamp if not provided
        if output_path is None:
            session_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"captions_{session_time}.srt"

        self.output_path = output_path
        self._lock = threading.Lock()
        self._sequence = 0
        self._session_start = time.time()  # Wall-clock reference point
        self._entries = []                 # In-memory list of all entries

        # Create/clear the file at session start
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("")

        print(f"[SRT] Writing captions to: {self.output_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, text: str, start_time: float, end_time: float):
        """
        Add a caption entry to the SRT file.

        Args:
            text:       The transcribed text for this caption block.
            start_time: Start time in seconds since session began.
            end_time:   End time in seconds since session began.
        """
        with self._lock:
            self._sequence += 1
            entry = {
                "sequence": self._sequence,
                "start":    start_time,
                "end":      end_time,
                "text":     text.strip(),
            }
            self._entries.append(entry)
            self._write_entry(entry)

    def add_entry_now(self, text: str, duration: float = 4.0):
        """
        Convenience method — uses current wall-clock time to
        calculate start/end automatically. Call this right after
        Whisper returns a result.

        Args:
            text:     Transcribed text.
            duration: How long this caption should display (seconds).
        """
        end_time = time.time() - self._session_start
        start_time = max(0.0, end_time - duration)
        self.add_entry(text, start_time, end_time)

    def get_output_path(self) -> str:
        return self.output_path

    def get_entry_count(self) -> int:
        return self._sequence

    def finalize(self):
        """
        Call when the session ends. Rewrites the full file cleanly
        to ensure the final output is well-formed.
        """
        with self._lock:
            with open(self.output_path, "w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(self._format_entry(entry))
        print(f"[SRT] Session complete. {self._sequence} caption(s) saved to: {self.output_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_entry(self, entry: dict):
        """Append a single entry to the file immediately."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(self._format_entry(entry))

    def _format_entry(self, entry: dict) -> str:
        """Format one SRT block as a string."""
        start_str = self._seconds_to_srt_time(entry["start"])
        end_str   = self._seconds_to_srt_time(entry["end"])
        return (
            f"{entry['sequence']}\n"
            f"{start_str} --> {end_str}\n"
            f"{entry['text']}\n"
            f"\n"
        )

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """
        Convert float seconds to SRT timestamp format.
        Example: 93.75 -> '00:01:33,750'
        """
        seconds = max(0.0, seconds)
        millis  = int((seconds % 1) * 1000)
        s       = int(seconds) % 60
        m       = (int(seconds) // 60) % 60
        h       = int(seconds) // 3600
        return f"{h:02}:{m:02}:{s:02},{millis:03}"


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running SRT exporter test...")

    exporter = SRTExporter(output_path="test_output.srt")

    # Simulate transcription results coming in over time
    test_captions = [
        "Hello, this is a live caption test.",
        "Real-time transcription powered by OpenAI Whisper.",
        "This file can be opened in VLC or imported into any video editor.",
        "Broadcast-quality subtitles generated automatically.",
        "Session complete.",
    ]

    for i, caption in enumerate(test_captions):
        start = i * 4.5
        end   = start + 4.0
        exporter.add_entry(caption, start, end)
        print(f"  Added entry {i + 1}: {caption[:50]}")
        time.sleep(0.2)

    exporter.finalize()

    # Print the resulting file
    print("\n--- Generated SRT file ---")
    with open("test_output.srt", "r", encoding="utf-8") as f:
        print(f.read())
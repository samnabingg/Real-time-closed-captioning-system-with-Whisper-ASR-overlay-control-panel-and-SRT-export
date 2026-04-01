"""
whisper_worker.py — faster-whisper + silero-VAD transcription pipeline
-----------------------------------------------------------------------
Two parallel workers:
    1. Buffer worker  — drains audio_queue into ring_buffer continuously
    2. VAD worker     — watches ring_buffer, detects speech segments,
                        transcribes with faster-whisper

Fix: Uses wall-clock time to track what has been transcribed, not
buffer-relative timestamps which reset every cycle.
"""

import time
import threading
import numpy as np
import torch
import asyncio
from scipy.signal import resample_poly
from math import gcd
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
from audio_streaming import (
    audio_queue, stream,
    DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE,
    ring_buffer, ring_buffer_lock
)
from srt_exporter import SRTExporter

# Set to True to see detailed pipeline decisions, False for clean output
DIAGNOSTIC_MODE = False  # Set True for debug

# --- Load faster-whisper model ---
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Whisper] Loading faster-whisper 'tiny' model on {DEVICE} ({COMPUTE_TYPE})... [Real-time optimized]")
model = WhisperModel("tiny", device=DEVICE, compute_type=COMPUTE_TYPE)
print("[Whisper] Model loaded.")

# --- Load silero-VAD ---
print("[VAD] Loading silero-VAD...")
vad_model = load_silero_vad()
print("[VAD] VAD loaded.")

# --- Tunable parameters (Latency-Optimized for 2-4s sweet spot) ---
SILENCE_THRESHOLD   = 0.005
MIN_SPEECH_SECONDS  = 1.0  # Skip short fillers/uh
MAX_SEGMENT_SECONDS = 6     # Slight increase for better context
VAD_THRESHOLD       = 0.4   # Stricter for cleaner speech
PADDING_SECONDS     = 0.3
VAD_CHECK_INTERVAL  = 0.25   # Check every 250ms for faster detection


def resample_to_16k(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def is_hallucination(text: str) -> bool:
    KNOWN = {"you", "thank you", "bye", "bye.", "you.", "thank you.", "", "hello", "hi", "um", "uh"}
    if text.lower().strip() in KNOWN:
        return True
    words = text.strip().split()
    if len(words) >= 6:
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) > 0.6:
            return True
    return False


def log_transcription(text: str, log_file: str = "transcription_log.txt"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


def diag(msg: str):
    if DIAGNOSTIC_MODE:
        print(f"  [DIAG] {msg}")


def buffer_worker(stop_event: threading.Event):
    """
    Worker 1 — continuously drains audio_queue into ring_buffer.
    Never pauses — always capturing even during inference.
    Each chunk is stored as a tuple of (chunk, wall_clock_time)
    so we can track absolute time position of each chunk.
    """
    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.05)
            with ring_buffer_lock:
                # Store chunk alongside the wall-clock time it arrived
                ring_buffer.append((chunk.flatten(), time.time()))
        except Exception:
            continue


def vad_transcribe_worker(
    stop_event: threading.Event,
    srt: SRTExporter,
    caption_display=None,
    control_panel=None,
):
    """
    Worker 2 — VAD + transcription.

    Key fix: each chunk in ring_buffer now carries its wall-clock arrival
    time. We track last_transcribed_wall_time (absolute) instead of a
    buffer-relative offset that resets every cycle.
    """
    # Wall-clock time of the last audio sample we successfully transcribed
    # Initialized to 0 so everything is new at startup
    last_transcribed_wall_time = 0.0
    cycle = 0

    while not stop_event.is_set():
        time.sleep(VAD_CHECK_INTERVAL)
        cycle += 1

        # Respect control panel start/stop
        if control_panel and not control_panel.is_running:
            # Reset wall time when stopped so next session starts fresh
            last_transcribed_wall_time = 0.0
            continue

        # Snapshot the ring buffer — list of (chunk, wall_time) tuples
        with ring_buffer_lock:
            if len(ring_buffer) == 0:
                diag("Ring buffer empty")
                continue
            snapshot = list(ring_buffer)

        # Split into audio arrays and wall times
        chunks     = [item[0] for item in snapshot]
        wall_times = [item[1] for item in snapshot]

        buffer_start_wall = wall_times[0]   # Wall time of oldest chunk
        buffer_end_wall   = wall_times[-1]  # Wall time of newest chunk
        buffer_seconds    = len(chunks) * 0.1

        # Concatenate audio (still at device sample rate)
        raw_audio = np.concatenate(chunks, axis=0).astype(np.float32)
        rms = np.sqrt(np.mean(raw_audio ** 2))

        diag(f"Cycle {cycle} | Buffer: {buffer_seconds:.1f}s | RMS: {rms:.4f} | Wall: {buffer_start_wall:.1f}→{buffer_end_wall:.1f}")

        if rms < SILENCE_THRESHOLD:
            diag(f"Skipped — RMS below threshold")
            continue

        # Resample to 16kHz
        audio_16k = resample_to_16k(raw_audio, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        audio_16k = normalize_audio(audio_16k)
        audio_16k = np.clip(audio_16k, -1.0, 1.0)

        # Run silero-VAD
        audio_tensor = torch.from_numpy(audio_16k)
        try:
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                vad_model,
                threshold=VAD_THRESHOLD,
                sampling_rate=WHISPER_SAMPLE_RATE,
                min_speech_duration_ms=int(MIN_SPEECH_SECONDS * 1000),
                speech_pad_ms=int(PADDING_SECONDS * 1000),
            )
        except Exception as e:
            print(f"[VAD error]: {e}")
            continue

        diag(f"VAD found {len(speech_timestamps)} segment(s)")

        if not speech_timestamps:
            continue

        new_segments = 0

        for i, segment in enumerate(speech_timestamps):
            start_sample = segment['start']
            end_sample   = segment['end']

            # Convert buffer-relative sample indices → wall-clock times
            # by mapping position in buffer to the wall_times array
            total_samples_16k = len(audio_16k)
            start_ratio = start_sample / total_samples_16k
            end_ratio   = end_sample   / total_samples_16k

            seg_start_wall = buffer_start_wall + start_ratio * buffer_seconds
            seg_end_wall   = buffer_start_wall + end_ratio   * buffer_seconds
            duration       = (end_sample - start_sample) / WHISPER_SAMPLE_RATE

            diag(f"  Seg {i+1}: wall {seg_start_wall:.2f}→{seg_end_wall:.2f} ({duration:.2f}s) | last_done={last_transcribed_wall_time:.2f}")

            # Skip if we already transcribed audio up to or past this segment
            if seg_end_wall <= last_transcribed_wall_time:
                diag(f"  Seg {i+1} skipped — already transcribed")
                continue

            # Skip too-short segments
            if duration < MIN_SPEECH_SECONDS:
                diag(f"  Seg {i+1} skipped — too short ({duration:.2f}s)")
                continue

            # If this segment overlaps with already-transcribed audio,
            # trim its start to only the new portion
            if seg_start_wall < last_transcribed_wall_time:
                overlap_seconds = last_transcribed_wall_time - seg_start_wall
                overlap_samples = int(overlap_seconds * WHISPER_SAMPLE_RATE)
                start_sample = start_sample + overlap_samples
                diag(f"  Seg {i+1} trimmed by {overlap_seconds:.2f}s overlap")

            # Cap very long segments
            if duration > MAX_SEGMENT_SECONDS:
                start_sample = end_sample - int(MAX_SEGMENT_SECONDS * WHISPER_SAMPLE_RATE)
                diag(f"  Seg {i+1} capped to {MAX_SEGMENT_SECONDS}s")

            speech_audio = audio_16k[start_sample:end_sample]
            if len(speech_audio) == 0:
                continue

            language = control_panel.selected_language if control_panel else None

            if caption_display:
                caption_display.set_status("Transcribing...")

            t_start = time.time()
            try:
                segments_iter, info = model.transcribe(
                    speech_audio,
                    language=language,
                    beam_size=3,
                    best_of=3,
                    temperature=(0.0, 0.2),
                    condition_on_previous_text=True,
                    vad_filter=True,
                    word_timestamps=False,
                )
                text = " ".join(seg.text for seg in segments_iter).strip()
                detected_lang = info.language
            except Exception as e:
                print(f"[Transcription error]: {e}")
                continue

            latency = time.time() - t_start

            # Advance wall-clock tracker to end of this segment
            last_transcribed_wall_time = seg_end_wall
            new_segments += 1

            diag(f"  Transcribed in {latency:.2f}s | '{text}'")

            if not text or is_hallucination(text):
                diag(f"  Hallucination filtered: '{text[:60]}'")
                continue

            print(f"[{time.strftime('%H:%M:%S')}] {text}  (latency: {latency:.2f}s)")

            log_transcription(text)
            srt.add_entry_now(text, duration=duration)

            if caption_display:
                caption_display.add_text(text)
                caption_display.set_status(
                    f"Listening...  |  {srt.get_entry_count()} captions  |  {latency:.2f}s"
                )

            if control_panel:
                control_panel.increment_caption_count(detected_lang)

        if DIAGNOSTIC_MODE and new_segments == 0:
            diag("No new segments this cycle")


async def transcribe_worker(caption_display=None, control_panel=None):
    """
    Main entry point — starts both workers on threads and waits.
    """
    srt = SRTExporter()
    stop_event = threading.Event()

    buf_thread = threading.Thread(
        target=buffer_worker,
        args=(stop_event,),
        daemon=True,
        name="BufferWorker"
    )

    vad_thread = threading.Thread(
        target=vad_transcribe_worker,
        args=(stop_event, srt, caption_display, control_panel),
        daemon=True,
        name="VADWorker"
    )

    buf_thread.start()
    vad_thread.start()

    print("\n[Diagnostic mode ON — set DIAGNOSTIC_MODE = False to disable]\n" if DIAGNOSTIC_MODE else "")

    try:
        while True:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        buf_thread.join(timeout=2)
        vad_thread.join(timeout=2)
        srt.finalize()
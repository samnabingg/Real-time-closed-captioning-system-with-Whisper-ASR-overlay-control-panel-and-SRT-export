"""
whisper_worker.py — Real-Time Closed Captioning (fixed)
---------------------------------------------------------
Key fix: Whisper transcription runs in a ThreadPoolExecutor so the
audio collection loop NEVER blocks. Audio captured during transcription
is no longer lost, eliminating the cut-off speech problem.

Audio flow:
    audio_queue (filled by sounddevice callback, always running)
        │
        ▼
    collect_loop (asyncio, drains queue every 10ms, fills buffer)
        │  when buffer hits BUFFER_SECONDS and no transcription running:
        ▼
    transcribe_in_thread (ThreadPoolExecutor — doesn't block collect_loop)
        │
        ▼
    caption_display / control_panel / SRT / eval
"""

import whisper
import time
import torch
import numpy as np
import asyncio
import concurrent.futures
import re
from scipy.signal import resample_poly
from math import gcd
from audio_streaming import audio_queue, stream, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE
from srt_exporter import SRTExporter

model = whisper.load_model("base")

SILENCE_THRESHOLD = 0.01
BUFFER_SECONDS    = 5        # seconds of audio per transcription chunk


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers (unchanged from your original)
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_16k(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    divisor = gcd(orig_sr, target_sr)
    up   = target_sr // divisor
    down = orig_sr   // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def is_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < threshold


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def is_hallucination(text: str) -> bool:
    KNOWN = {"you", "thank you", "bye", "bye.", "you.", "thank you.", ""}
    clean = text.lower().strip()
    if clean in KNOWN:
        return True
    # URL/email hallucinations
    if re.search(r"(https?://|www\.|\.com|@)", clean):
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


# ─────────────────────────────────────────────────────────────────────────────
# Transcription (runs in thread — does NOT block the audio loop)
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_sync(audio_data: np.ndarray, language=None) -> tuple[str, str, float]:
    """
    Pure synchronous transcription — safe to run in a thread.
    Returns (text, detected_lang, latency_seconds).
    """
    audio_data = resample_to_16k(audio_data, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
    audio_data = normalize_audio(audio_data)
    audio_data = np.clip(audio_data, -1.0, 1.0)

    audio_padded = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)

    options = whisper.DecodingOptions(
        language=language,
        fp16=torch.cuda.is_available(),
        without_timestamps=True,
        suppress_tokens=[-1],
    )

    start  = time.time()
    result = whisper.decode(model, mel, options)
    latency = time.time() - start

    text          = result.text.strip()
    detected_lang = getattr(result, "language", None)
    return text, detected_lang, latency


# ─────────────────────────────────────────────────────────────────────────────
# Output handler (called from the asyncio loop after transcription finishes)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_result(text, detected_lang, latency, srt, caption_display, control_panel):
    """Handle a completed transcription result — update UI, SRT, eval."""
    if is_hallucination(text):
        short = text[:60] + ("…" if len(text) > 60 else "")
        print(f"[Skipped hallucination]: '{short}'")
        return

    print(f"[{time.strftime('%H:%M:%S')}] {text}  (latency: {latency:.2f}s)")
    log_transcription(text)
    srt.add_entry_now(text, duration=BUFFER_SECONDS + latency)

    if caption_display:
        caption_display.add_text(text)
        caption_display.set_status(
            f"Listening...  |  {srt.get_entry_count()} captions  |  {latency:.2f}s"
        )

    # ── Accuracy eval ─────────────────────────────────────────────────
    if (control_panel
            and getattr(control_panel, "eval_enabled", None)
            and control_panel.eval_enabled.get()):

        reference = (control_panel.consume_queued_reference()
                     if hasattr(control_panel, "consume_queued_reference") else None)

        if reference is None:
            if hasattr(control_panel, "eval_last_result_var"):
                control_panel.eval_last_result_var.set(
                    "Waiting for reference (click Submit for next)"
                )
        else:
            from accuracy_eval import AccuracySession
            if not hasattr(control_panel, "_accuracy_session") or \
               control_panel._accuracy_session is None:
                control_panel._accuracy_session = AccuracySession()

            r = control_panel._accuracy_session.add(
                reference=reference,
                hypothesis=text,
                latency_sec=latency,
            )
            acc_pct = AccuracySession.accuracy_pct_from_wer(r.wer)
            control_panel.eval_last_result_var.set(
                f"Acc: {acc_pct:.1f}% | WER: {r.wer:.3f} | Lat: {r.latency_sec:.2f}s"
            )
            if caption_display:
                caption_display.set_status(
                    f"Acc: {acc_pct:.1f}%  |  Lat: {r.latency_sec:.2f}s"
                )

    if control_panel:
        control_panel.increment_caption_count(detected_lang)


# ─────────────────────────────────────────────────────────────────────────────
# Main worker — audio collection loop never blocks
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe_worker(caption_display=None, control_panel=None):
    """
    Audio collection and transcription run concurrently.

    The collect loop drains audio_queue every 10ms — always.
    When the buffer is full, transcription is fired off in a
    ThreadPoolExecutor and the collect loop keeps running immediately.
    Audio captured during transcription is NOT lost.
    """
    buffer        = []
    silent_chunks = 0
    srt           = SRTExporter()
    was_running   = False
    transcribing  = False   # flag: is a transcription currently in flight?

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop     = asyncio.get_event_loop()

    async def _fire_transcription(audio_snapshot):
        """Launch transcription in thread, handle result when done."""
        nonlocal transcribing
        language = control_panel.selected_language if control_panel else None
        try:
            text, detected_lang, latency = await loop.run_in_executor(
                executor,
                _transcribe_sync,
                audio_snapshot,
                language,
            )
            _handle_result(text, detected_lang, latency,
                           srt, caption_display, control_panel)
        except Exception as e:
            print(f"[Transcription error]: {e}")
        finally:
            transcribing = False

    try:
        while True:
            is_running = control_panel.is_running if control_panel else True

            if not is_running:
                if was_running:
                    srt.finalize()
                    srt = SRTExporter()
                    buffer        = []
                    silent_chunks = 0
                    transcribing  = False
                was_running = False
                await asyncio.sleep(0.1)
                continue

            was_running = True

            # ── Drain the audio queue ──────────────────────────────────
            drained = 0
            while not audio_queue.empty() and drained < 10:
                chunk = audio_queue.get().flatten()
                drained += 1

                if is_silence(chunk):
                    silent_chunks += 1
                    if silent_chunks > 3:
                        # Only clear the visual status — don't lose buffered audio
                        if caption_display:
                            caption_display.set_status("Listening...")
                else:
                    silent_chunks = 0
                    buffer.append(chunk)

            # ── Fire transcription when buffer is full and none in flight ──
            if len(buffer) >= BUFFER_SECONDS and not transcribing:
                transcribing = True

                # Snapshot the buffer and clear immediately so new audio
                # starts accumulating right away — no gap, no cut-off
                audio_snapshot = np.concatenate(buffer, axis=0)
                buffer = []

                if caption_display:
                    caption_display.set_status("Transcribing...")

                # Non-blocking — collect loop continues while this runs
                asyncio.ensure_future(_fire_transcription(audio_snapshot))

            await asyncio.sleep(0.01)   # 10ms tick — tight enough to not drop chunks

    except asyncio.CancelledError:
        pass
    finally:
        executor.shutdown(wait=False)
        srt.finalize()
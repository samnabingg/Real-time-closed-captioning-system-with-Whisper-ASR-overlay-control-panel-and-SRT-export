import whisper
import time
import torch
import numpy as np
import asyncio
from scipy.signal import resample_poly
from math import gcd
from audio_streaming import audio_queue, stream, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE
from srt_exporter import SRTExporter

model = whisper.load_model("base")

SILENCE_THRESHOLD = 0.01
BUFFER_SECONDS = 5


def resample_to_16k(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
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
    KNOWN_HALLUCINATIONS = {"you", "thank you", "bye", "bye.", "you.", "thank you.", ""}
    if text.lower().strip() in KNOWN_HALLUCINATIONS:
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


async def transcribe_worker(caption_display=None, control_panel=None):
    """
    Main transcription loop.
    - Respects control_panel.is_running flag (Start/Stop)
    - Uses control_panel.selected_language for Whisper language
    - Sends text to caption overlay
    - Writes SRT file per session
    - Updates control panel stats
    """
    buffer = []
    silent_chunks = 0
    srt = SRTExporter()
    was_running = False

    try:
        while True:
            # Respect the Start/Stop toggle from the control panel
            is_running = control_panel.is_running if control_panel else True

            if not is_running:
                # If we just stopped, flush and finalize
                if was_running:
                    srt.finalize()
                    srt = SRTExporter()   # Fresh SRT ready for next session
                    buffer = []
                    silent_chunks = 0
                was_running = False
                await asyncio.sleep(0.1)
                continue

            was_running = True

            if not audio_queue.empty():
                chunk = audio_queue.get().flatten()

                if is_silence(chunk):
                    silent_chunks += 1
                    if silent_chunks > 3:
                        buffer = []
                        if caption_display:
                            caption_display.set_status("Listening...")
                    continue
                else:
                    silent_chunks = 0
                    buffer.append(chunk)

                if len(buffer) >= BUFFER_SECONDS:
                    if caption_display:
                        caption_display.set_status("Transcribing...")

                    audio_data = np.concatenate(buffer, axis=0)
                    audio_data = resample_to_16k(audio_data, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
                    audio_data = normalize_audio(audio_data)
                    audio_data = np.clip(audio_data, -1.0, 1.0)

                    audio_padded = whisper.pad_or_trim(audio_data)
                    mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)

                    # Pick up language from control panel (None = auto-detect)
                    language = control_panel.selected_language if control_panel else None

                    options = whisper.DecodingOptions(
                        language=language,
                        fp16=torch.cuda.is_available(),
                        without_timestamps=True,
                        suppress_tokens=[-1],
                    )

                    start = time.time()
                    result = whisper.decode(model, mel, options)
                    latency = time.time() - start

                    text = result.text.strip()

                    # Detected language comes back on the result object
                    detected_lang = getattr(result, "language", None)

                    if is_hallucination(text):
                        print(f"[Skipped hallucination]: '{text[:60]}'" if len(text) > 60 else f"[Skipped hallucination]: '{text}'")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] {text}  (latency: {latency:.2f}s)")

                        log_transcription(text)
                        srt.add_entry_now(text, duration=BUFFER_SECONDS + latency)

                        if caption_display:
                            caption_display.add_text(text)
                            caption_display.set_status(
                                f"Listening...  |  {srt.get_entry_count()} captions  |  {latency:.2f}s"
                            )

                        if control_panel:
                            control_panel.increment_caption_count(detected_lang)

                    buffer = []

            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass
    finally:
        srt.finalize()
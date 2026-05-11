"""
server.py — VoyaVox Live Captions  |  Link Electronics Demo
------------------------------------------------------------
FastAPI WebSocket backend. Browser sends raw Float32 PCM chunks;
server resamples → Whisper → sends caption JSON back.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Open:
    http://localhost:8000
"""

import asyncio
import concurrent.futures
import json
import re
import time

import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from math import gcd

try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="VoyaVox Live Captions")

print("\n" + "=" * 52)
print("  VoyaVox — Real-Time Captions  |  Link Electronics")
print("=" * 52)
print("  Loading Whisper base model...", end="", flush=True)
model = whisper.load_model("base")
print(" ✓ ready\n")

WHISPER_SR        = 16000
SILENCE_THRESHOLD = 0.008
BUFFER_SECONDS    = 5        # seconds of audio before transcribing

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == WHISPER_SR:
        return audio
    if HAS_SCIPY:
        d  = gcd(orig_sr, WHISPER_SR)
        return resample_poly(audio, WHISPER_SR // d, orig_sr // d).astype(np.float32)
    # Fallback: naive linear interpolation
    ratio   = WHISPER_SR / orig_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def is_hallucination(text: str) -> bool:
    """Filter common Whisper hallucinations."""
    KNOWN = {"you", "thank you", "bye", "bye.", "you.", "thank you.", ""}
    clean = text.lower().strip()
    if clean in KNOWN:
        return True
    if re.search(r"(https?://|www\.|\.com|@)", clean):
        return True
    words = text.strip().split()
    if len(words) >= 6:
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) > 0.6:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Transcription (synchronous — runs in thread pool)
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_sync(audio: np.ndarray, language: str | None) -> dict:
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)

    t0     = time.time()
    result = model.transcribe(
        audio,
        language=language,
        fp16=False,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    latency = time.time() - t0

    return {
        "text":     result["text"].strip(),
        "language": result.get("language"),
        "latency":  round(latency, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    buffer: list[np.ndarray] = []
    transcribing              = False
    client_sr                 = WHISPER_SR   # updated by hello handshake
    selected_language: str | None = None

    print("[WS] Client connected")

    try:
        while True:
            msg = await websocket.receive()

            # ── Text message (control / handshake) ─────────────────────────
            if "text" in msg:
                data = json.loads(msg["text"])

                if data["type"] == "hello":
                    client_sr         = data.get("sampleRate", WHISPER_SR)
                    selected_language = data.get("language") or None
                    print(f"[WS] hello — sr={client_sr}  lang={selected_language or 'auto'}")
                    await websocket.send_json({"type": "ready"})

                elif data["type"] == "language":
                    selected_language = data.get("language") or None
                    print(f"[WS] language → {selected_language or 'auto'}")

                elif data["type"] == "stop":
                    buffer.clear()
                    transcribing = False
                    await websocket.send_json({"type": "status", "text": "Stopped"})

            # ── Binary message (Float32 PCM audio) ─────────────────────────
            elif "bytes" in msg:
                raw      = np.frombuffer(msg["bytes"], dtype=np.float32)
                audio_16k = resample_to_16k(raw, client_sr)

                # Only buffer non-silent audio
                rms = np.sqrt(np.mean(audio_16k ** 2))
                if rms > SILENCE_THRESHOLD:
                    buffer.append(audio_16k)

                # Fire when buffer holds enough speech
                total = sum(a.shape[0] for a in buffer)
                if total >= WHISPER_SR * BUFFER_SECONDS and not transcribing:
                    transcribing   = True
                    snapshot       = np.concatenate(buffer, axis=0)
                    buffer.clear()

                    await websocket.send_json({"type": "status", "text": "Transcribing…"})

                    async def _run(snap=snapshot, lang=selected_language):
                        nonlocal transcribing
                        try:
                            result = await loop.run_in_executor(
                                executor, _transcribe_sync, snap, lang
                            )
                            text = result["text"]
                            if text and not is_hallucination(text):
                                await websocket.send_json({
                                    "type":     "caption",
                                    "text":     text,
                                    "language": result["language"],
                                    "latency":  result["latency"],
                                })
                            await websocket.send_json({
                                "type": "status", "text": "Listening…"
                            })
                        except Exception as e:
                            print(f"[Transcription error] {e}")
                        finally:
                            transcribing = False

                    asyncio.ensure_future(_run())

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            print("[WS] Client disconnected (runtime)")
        else:
            print(f"[WS error] {e}")
    except Exception as e:
        print(f"[WS error] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Static files (index.html lives in ./static/)
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")
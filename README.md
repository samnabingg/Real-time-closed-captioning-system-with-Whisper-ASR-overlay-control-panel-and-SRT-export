# Real-Time Speech Transcription with OpenAI Whisper

A lightweight, real-time audio transcription system built on top of OpenAI's Whisper ASR model. Captures live microphone input, buffers audio, resamples it to Whisper's required format, and outputs transcribed text — with automatic hallucination filtering and timestamped logging.

---

## Features

- **Real-time audio capture** using `sounddevice` with device-native sample rates
- **Automatic resampling** from device sample rate (e.g. 48kHz) down to Whisper's required 16kHz using polyphase filtering (`scipy`)
- **Silence detection** — skips transcription on quiet/empty audio to reduce noise and hallucination
- **Hallucination filtering** — detects and discards looping token artifacts common in low-signal ASR output
- **Auto language detection** — no need to hardcode a language; Whisper identifies it automatically
- **Persistent transcription log** — all valid transcriptions are timestamped and saved to `transcription_log.txt`
- **Low-latency pipeline** — async architecture with `asyncio` keeps the audio stream and transcription loop non-blocking

---

## Architecture

```
Microphone Input (sounddevice)
        │
        ▼
  Audio Callback → audio_queue (thread-safe)
        │
        ▼
  Buffer 5 seconds of speech chunks
        │
        ▼
  Silence check → skip if RMS < threshold
        │
        ▼
  Resample: device_rate → 16kHz (polyphase)
        │
        ▼
  Normalize + clip to [-1.0, 1.0]
        │
        ▼
  Whisper: pad_or_trim → mel spectrogram → decode
        │
        ▼
  Hallucination filter → print + log
```

---

## Requirements

```
openai-whisper
sounddevice
numpy
scipy
torch
```

Install with:

```bash
pip install openai-whisper sounddevice numpy scipy torch
```

> **Note:** On Windows, you may also need [FFmpeg](https://ffmpeg.org/) on your PATH for Whisper's audio backend.

---

## Setup

**1. Check your microphone device index:**

```bash
python audio_streaming.py
```

This prints all available audio devices. Find your microphone's index and update `DEVICE_INDEX` in `audio_streaming.py`.

**2. Run the transcription worker:**

```bash
python whisper_worker.py
```

---

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `DEVICE_INDEX` | `audio_streaming.py` | `9` | Microphone device index |
| `CHUNK_DURATION` | `audio_streaming.py` | `1.0s` | Size of each audio chunk |
| `BUFFER_SECONDS` | `whisper_worker.py` | `5` | Seconds to buffer before transcribing |
| `SILENCE_THRESHOLD` | `whisper_worker.py` | `0.01` | RMS threshold below which audio is skipped |
| `LOG_FILE` | `whisper_worker.py` | `transcription_log.txt` | Output log path |

---

## Test Video

[![Watch the video](https://img.youtube.com/vi/n1JCyEHqI-M/0.jpg)](https://www.youtube.com/watch?v=n1JCyEHqI-M)

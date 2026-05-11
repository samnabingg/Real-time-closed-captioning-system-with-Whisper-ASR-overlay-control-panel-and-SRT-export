"""
realtime_eval.py — Live Accuracy Evaluator
===========================================
Speak a sentence, type what you said, get your WER/accuracy score instantly.

HOW IT WORKS:
─────────────
1. Press ENTER to start recording
2. Speak your sentence clearly
3. Press ENTER again to stop
4. Type what you actually said (the "reference")
5. See your WER, accuracy %, and latency instantly
6. Repeat as many times as you want
7. Type 'report' to see your full session summary
8. Type 'quit' to exit

REQUIREMENTS:
─────────────
    pip install openai-whisper sounddevice numpy scipy jiwer

Run:
    python realtime_eval.py
    python realtime_eval.py --model small     (more accurate, slower)
    python realtime_eval.py --model medium    (most accurate)
"""

import argparse
import time
import sys
import re
import numpy as np
import sounddevice as sd
import whisper
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_SAMPLE_RATE = 16000
RECORD_SAMPLE_RATE  = 44100     # Most mics default to this
MAX_RECORD_SECONDS  = 30        # Safety cap

# ── Result store ──────────────────────────────────────────────────────────────
@dataclass
class LiveResult:
    take:         int
    reference:    str
    hypothesis:   str
    wer:          float
    cer:          float
    latency_sec:  float
    word_count:   int
    error_detail: dict = field(default_factory=dict)


session_results: list[LiveResult] = []
take_number = 0


# ─────────────────────────────────────────────────────────────────────────────
# Audio recording
# ─────────────────────────────────────────────────────────────────────────────

def record_until_enter() -> np.ndarray:
    """
    Records from mic until user presses ENTER.
    Returns float32 numpy array at RECORD_SAMPLE_RATE.
    """
    audio_chunks = []
    stop_event   = threading.Event()
    q            = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass  # suppress noise warnings
        q.put(indata.copy())

    def collect():
        while not stop_event.is_set():
            try:
                chunk = q.get(timeout=0.1)
                audio_chunks.append(chunk)
            except queue.Empty:
                continue

    stream = sd.InputStream(
        samplerate=RECORD_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(RECORD_SAMPLE_RATE * 0.1),
        callback=callback,
    )

    collector_thread = threading.Thread(target=collect, daemon=True)

    with stream:
        collector_thread.start()
        input()   # blocks until ENTER
        stop_event.set()

    collector_thread.join(timeout=1.0)

    if not audio_chunks:
        return np.zeros(RECORD_SAMPLE_RATE, dtype=np.float32)

    return np.concatenate(audio_chunks, axis=0).flatten()


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    if orig_sr == target_sr:
        return audio
    d = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // d, orig_sr // d).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy measurement
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def score(reference: str, hypothesis: str) -> tuple[float, float, dict]:
    from jiwer import wer, cer, process_words
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    w     = wer(ref_n, hyp_n)
    c     = cer(ref_n, hyp_n)
    m     = process_words(ref_n, hyp_n)
    detail = {
        "hits":          m.hits,
        "substitutions": m.substitutions,
        "deletions":     m.deletions,
        "insertions":    m.insertions,
    }
    return w, c, detail


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def color(text, code): return f"\033[{code}m{text}\033[0m"
green  = lambda t: color(t, "32")
yellow = lambda t: color(t, "33")
red    = lambda t: color(t, "31")
cyan   = lambda t: color(t, "36")
bold   = lambda t: color(t, "1")

def accuracy_bar(acc: float, width: int = 30) -> str:
    filled = int(acc / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    if acc >= 95:   bar_colored = green(bar)
    elif acc >= 85: bar_colored = yellow(bar)
    else:           bar_colored = red(bar)
    return f"[{bar_colored}] {acc:.1f}%"

def print_result(result: LiveResult):
    acc = (1 - result.wer) * 100
    print()
    print(bold("─" * 58))
    print(f"  Take #{result.take}")
    print(bold("─" * 58))

    # Show word-level diff
    ref_words = normalize(result.reference).split()
    hyp_words = normalize(result.hypothesis).split()
    print(f"\n  {cyan('You said (reference):')}")
    print(f"    {result.reference}")
    print(f"\n  {cyan('Whisper heard:')}")

    # Simple token alignment highlight
    hyp_display = []
    hyp_set = set(hyp_words)
    for w in hyp_words:
        if w in ref_words:
            hyp_display.append(green(w))
        else:
            hyp_display.append(red(w))
    print(f"    {' '.join(hyp_display)}")

    print(f"\n  {cyan('Accuracy:')}  {accuracy_bar(acc)}")
    print(f"  WER:        {result.wer:.3f}   CER: {result.cer:.3f}")
    print(f"  Latency:    {result.latency_sec:.2f}s   |   Words: {result.word_count}")
    print(f"  Errors:     {result.error_detail['substitutions']} substitutions  "
          f"{result.error_detail['deletions']} deletions  "
          f"{result.error_detail['insertions']} insertions")

    # Voyavox comparison
    print()
    if acc >= 95:
        print(f"  {green('✓ Matches Voyavox 95% accuracy target')}")
    elif acc >= 90:
        print(f"  {yellow('~ Close to Voyavox 95% target — try the small model')}")
    else:
        print(f"  {red('✗ Below Voyavox target — run with --model small')}")

    if result.latency_sec < 5.0:
        print(f"  {green(f'✓ Beats Voyavox <5s latency target ({result.latency_sec:.2f}s)')}")
    else:
        print(f"  {red(f'✗ Over Voyavox 5s latency target ({result.latency_sec:.2f}s)')}")
    print(bold("─" * 58))


def print_session_report():
    if not session_results:
        print("\n  No results yet.\n")
        return

    import json
    avg_wer  = np.mean([r.wer for r in session_results])
    avg_lat  = np.mean([r.latency_sec for r in session_results])
    avg_acc  = (1 - avg_wer) * 100
    best     = min(session_results, key=lambda r: r.wer)
    worst    = max(session_results, key=lambda r: r.wer)

    print()
    print(bold("═" * 58))
    print(bold("  SESSION REPORT"))
    print(bold("═" * 58))
    print(f"  Takes recorded    : {len(session_results)}")
    print(f"  Avg word accuracy : {accuracy_bar(avg_acc)}")
    print(f"  Avg WER           : {avg_wer:.3f}")
    print(f"  Avg latency       : {avg_lat:.2f}s")
    print(f"  Best take         : #{best.take}  ({(1-best.wer)*100:.1f}%)")
    print(f"  Hardest take      : #{worst.take} ({(1-worst.wer)*100:.1f}%)")
    print()

    print(f"  {'Take':<6} {'Accuracy':>10} {'WER':>7} {'Latency':>9}  Reference")
    print("  " + "─" * 54)
    for r in session_results:
        acc = (1 - r.wer) * 100
        ref_short = r.reference[:30] + ("…" if len(r.reference) > 30 else "")
        marker = green("●") if acc >= 95 else yellow("●") if acc >= 85 else red("●")
        print(f"  {marker} #{r.take:<4} {acc:>9.1f}% {r.wer:>7.3f} {r.latency_sec:>8.2f}s  {ref_short}")

    print()
    voyavox_passes = sum(1 for r in session_results if (1-r.wer)*100 >= 95)
    print(f"  Voyavox 95% target: {voyavox_passes}/{len(session_results)} takes passed")
    latency_passes = sum(1 for r in session_results if r.latency_sec < 5.0)
    print(f"  Voyavox <5s target: {latency_passes}/{len(session_results)} takes passed")

    # Save JSON
    path = f"realtime_eval_session_{time.strftime('%Y%m%d_%H%M%S')}.json"
    data = [
        {
            "take": r.take,
            "reference": r.reference,
            "hypothesis": r.hypothesis,
            "wer": round(r.wer, 4),
            "accuracy_pct": round((1-r.wer)*100, 2),
            "latency_sec": round(r.latency_sec, 3),
        }
        for r in session_results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(bold("═" * 58))
    print(f"\n  Full session saved → {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global take_number

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base",
                        help="Whisper model: tiny / base / small / medium (default: base)")
    args = parser.parse_args()

    print()
    print(bold("═" * 58))
    print(bold("  Real-Time Whisper Accuracy Evaluator"))
    print(bold("═" * 58))
    print(f"  Model   : {args.model}")
    print(f"  Mic rate: {RECORD_SAMPLE_RATE}Hz → resampled to {WHISPER_SAMPLE_RATE}Hz")
    print()
    print("  Commands:  'report' — session summary")
    print("             'quit'  — exit")
    print(bold("═" * 58))

    print(f"\n  Loading whisper '{args.model}' model...", end="", flush=True)
    model = whisper.load_model(args.model)
    print(green(" ready"))

    while True:
        print()
        cmd = input(bold("  Press ENTER to record  (or type 'report' / 'quit'): ")).strip().lower()

        if cmd == "quit":
            print_session_report()
            print("  Goodbye!\n")
            break

        if cmd == "report":
            print_session_report()
            continue

        # ── Record ────────────────────────────────────────────────────────────
        take_number += 1
        print(f"\n  {green('● Recording...')}  (speak now, then press ENTER to stop)")

        raw_audio = record_until_enter()

        if len(raw_audio) < RECORD_SAMPLE_RATE * 0.3:
            print(f"  {yellow('Too short — try again')}")
            take_number -= 1
            continue

        duration = len(raw_audio) / RECORD_SAMPLE_RATE
        print(f"  Recorded {duration:.1f}s of audio")

        # ── Transcribe ────────────────────────────────────────────────────────
        print(f"  {cyan('Transcribing...')}", end="", flush=True)
        audio_16k = resample(raw_audio, RECORD_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        audio_16k = np.clip(audio_16k / (np.max(np.abs(audio_16k)) + 1e-8), -1.0, 1.0)

        t0     = time.time()
        result = model.transcribe(
            audio_16k,
            fp16=False,
            without_timestamps=True,
            language=None,   # auto-detect
        )
        latency    = time.time() - t0
        hypothesis = result["text"].strip()
        print(f" done ({latency:.2f}s)")
        print(f"\n  Whisper heard: {cyan(hypothesis)}")

        # ── Get reference ─────────────────────────────────────────────────────
        print()
        reference = input("  Type exactly what you said (reference): ").strip()

        if not reference:
            print(f"  {yellow('Skipped — no reference entered')}")
            take_number -= 1
            continue

        # ── Score ─────────────────────────────────────────────────────────────
        wer_score, cer_score, detail = score(reference, hypothesis)
        word_count = len(normalize(reference).split())

        live_result = LiveResult(
            take=take_number,
            reference=reference,
            hypothesis=hypothesis,
            wer=wer_score,
            cer=cer_score,
            latency_sec=latency,
            word_count=word_count,
            error_detail=detail,
        )
        session_results.append(live_result)
        print_result(live_result)


if __name__ == "__main__":
    main()
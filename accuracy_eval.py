"""
accuracy_eval.py — In-session accuracy tracking
------------------------------------------------
Used by whisper_worker.py to score each caption against
a reference the user types into the control panel.

Drop this file in the same folder as whisper_worker.py.
No other changes needed — the worker already imports it.
"""

import re
import time
import json
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TakeResult:
    take:        int
    reference:   str
    hypothesis:  str
    wer:         float
    cer:         float
    latency_sec: float
    timestamp:   str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


class AccuracySession:
    """
    Accumulates WER/CER scores across the live session.
    whisper_worker.py calls .add() each time eval is enabled
    and a reference has been queued via the control panel.
    """

    def __init__(self):
        self.takes: list[TakeResult] = []

    # ── public API ────────────────────────────────────────────────────

    def add(self, reference: str, hypothesis: str, latency_sec: float) -> TakeResult:
        """Score one caption and store the result. Returns the TakeResult."""
        from jiwer import wer, cer
        ref_n = self._normalize(reference)
        hyp_n = self._normalize(hypothesis)
        wer_score = wer(ref_n, hyp_n)
        cer_score = cer(ref_n, hyp_n)
        result = TakeResult(
            take=len(self.takes) + 1,
            reference=reference,
            hypothesis=hypothesis,
            wer=round(wer_score, 4),
            cer=round(cer_score, 4),
            latency_sec=round(latency_sec, 3),
        )
        self.takes.append(result)
        self._print_result(result)
        return result

    def avg_wer(self) -> float:
        if not self.takes:
            return 0.0
        return float(np.mean([t.wer for t in self.takes]))

    def avg_accuracy_pct(self) -> float:
        return self.accuracy_pct_from_wer(self.avg_wer())

    def avg_latency(self) -> float:
        if not self.takes:
            return 0.0
        return float(np.mean([t.latency_sec for t in self.takes]))

    def save_report(self, path: str = None) -> str:
        path = path or f"accuracy_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            "session_summary": {
                "takes": len(self.takes),
                "avg_wer": round(self.avg_wer(), 4),
                "avg_accuracy_pct": round(self.avg_accuracy_pct(), 2),
                "avg_latency_sec": round(self.avg_latency(), 3),
                "voyavox_95pct_target_met": self.avg_accuracy_pct() >= 95.0,
                "voyavox_5s_latency_met": self.avg_latency() < 5.0,
            },
            "takes": [
                {
                    "take": t.take,
                    "time": t.timestamp,
                    "reference": t.reference,
                    "hypothesis": t.hypothesis,
                    "wer": t.wer,
                    "accuracy_pct": round(self.accuracy_pct_from_wer(t.wer), 2),
                    "cer": t.cer,
                    "latency_sec": t.latency_sec,
                }
                for t in self.takes
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[AccuracySession] Report saved → {path}")
        return path

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def accuracy_pct_from_wer(wer_score: float) -> float:
        return max(0.0, (1.0 - wer_score) * 100)

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _print_result(self, r: TakeResult):
        acc = self.accuracy_pct_from_wer(r.wer)
        bar_len = 20
        filled = int(acc / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        voyavox = "✓ beats Voyavox 95%" if acc >= 95 else "✗ below Voyavox 95%"
        print(
            f"[Eval #{r.take}] {bar} {acc:.1f}%  "
            f"WER={r.wer:.3f}  CER={r.cer:.3f}  "
            f"Lat={r.latency_sec:.2f}s  {voyavox}"
        )
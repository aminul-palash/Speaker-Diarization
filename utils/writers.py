"""
utils/writers.py
Write transcript output in TXT, SRT and JSON formats.
"""

import json
from typing import List, Dict


# ── SRT timestamp helper ──────────────────────────────────────────────────

def _srt_time(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    ms     = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


# ── Writers ───────────────────────────────────────────────────────────────

def write_txt(
    utterances: List[dict],
    path: str,
) -> None:
    """
    Plain-text transcript.
    Format:
        [SPEAKER_00]
        আপনার কি সমস্যা হচ্ছে?

        [SPEAKER_01]
        জ্বর আছে তিন দিন ধরে।
    """
    with open(path, "w", encoding="utf-8") as f:
        for u in utterances:
            f.write(f"[{u['speaker']}]\n{u['text'].strip()}\n\n")
    print(f"   ✅ TXT  → {path}")


def write_srt(
    utterances: List[dict],
    path: str,
) -> None:
    """
    SRT subtitle file with speaker labels.
    """
    with open(path, "w", encoding="utf-8") as f:
        for i, u in enumerate(utterances, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_time(u['start'])} --> {_srt_time(u['end'])}\n")
            f.write(f"[{u['speaker']}]: {u['text'].strip()}\n\n")
    print(f"   ✅ SRT  → {path}")


def write_json(
    utterances: List[dict],
    path: str,
    meta: dict = None,
) -> None:
    """
    JSON output — machine-readable, ready for downstream
    clinical NLP / prescription generation.
    """
    payload  = {
        "meta": meta or {},
        "utterances": utterances,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"   ✅ JSON → {path}")

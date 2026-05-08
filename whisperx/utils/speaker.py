"""
utils/speaker.py
Speaker-to-word assignment, utterance grouping, and statistics.
"""

from collections import defaultdict
from typing import List, Dict


def assign_speakers_to_words(
    word_segments: List[dict],
    diarization
) -> List[dict]:
    """
    Map each word's midpoint timestamp to the active speaker
    from the Pyannote diarization result.
    Falls back to the nearest speaker segment when there is
    no exact overlap (handles very short gaps between turns).
    """
    turns = [
        (seg.start, seg.end, spk)
        for seg, _, spk in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda x: x[0])

    def speaker_at(t: float) -> str:
        for s, e, spk in turns:
            if s <= t <= e:
                return spk
        # nearest midpoint fallback
        return min(turns, key=lambda x: abs(t - (x[0] + x[1]) / 2))[2]

    result = []
    for w in word_segments:
        mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2
        result.append({**w, "speaker": speaker_at(mid)})
    return result


def group_into_utterances(word_speaker_list: List[dict]) -> List[dict]:
    """
    Merge consecutive words from the same speaker into
    utterance blocks: {speaker, start, end, text}.
    """
    if not word_speaker_list:
        return []

    utterances = []
    cur = {
        "speaker": word_speaker_list[0]["speaker"],
        "start":   word_speaker_list[0].get("start", 0.0),
        "end":     word_speaker_list[0].get("end",   0.0),
        "text":    word_speaker_list[0].get("word",  ""),
    }

    for w in word_speaker_list[1:]:
        if w["speaker"] == cur["speaker"]:
            cur["text"] += " " + w.get("word", "")
            cur["end"]   = w.get("end", cur["end"])
        else:
            utterances.append(cur)
            cur = {
                "speaker": w["speaker"],
                "start":   w.get("start", 0.0),
                "end":     w.get("end",   0.0),
                "text":    w.get("word",  ""),
            }
    utterances.append(cur)
    return utterances


def speaker_stats(utterances: List[dict]) -> Dict[str, dict]:
    """Return word count and total speaking time per speaker."""
    stats = defaultdict(lambda: {"words": 0, "seconds": 0.0})
    for u in utterances:
        spk = u["speaker"]
        stats[spk]["words"]   += len(u["text"].split())
        stats[spk]["seconds"] += u["end"] - u["start"]
    return dict(stats)


def heuristic_role_map(utterances: List[dict]) -> Dict[str, str]:
    """
    Assign roles based on word count:
      most words  → doctor
      second most → patient
      rest        → attendant

    This is a reasonable default for clinical audio where the
    doctor typically speaks the most. Override with the LLM
    classifier in utils/role_classifier.py for higher accuracy.
    """
    stats  = speaker_stats(utterances)
    ranked = sorted(stats.items(), key=lambda x: -x[1]["words"])
    labels = ["doctor", "patient"] + ["attendant"] * 20
    return {spk: labels[i] for i, (spk, _) in enumerate(ranked)}

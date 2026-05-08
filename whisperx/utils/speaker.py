"""
utils/speaker.py
Speaker-to-word assignment, utterance grouping, and statistics.
"""

from collections import defaultdict
from typing import List, Dict


def assign_speakers_to_words(
    word_segments: List[dict],
    diarization,
    gap_threshold: float = 1.0
) -> List[dict]:
    """
    Map each word to the active speaker from Pyannote diarization.
    
    Priority:
    1. Overlap with a speaker turn (word time range overlaps speaker turn)
    2. If word is within gap_threshold seconds of next speaker's turn, assign to next speaker
    3. Most recent speaker (whose turn ended closest before the word)
    4. Nearest speaker turn overall
    
    gap_threshold: seconds - if gap between speakers is <= this, words in gap
                   are assigned to the NEXT speaker (helps with timing misalignment)
    """
    turns = [
        (seg.start, seg.end, spk)
        for seg, _, spk in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda x: x[0])

    def speaker_at(w_start: float, w_end: float) -> str:
        # First: overlap with a speaker turn
        for s, e, spk in turns:
            # Check if word time range [w_start, w_end] overlaps with turn [s, e]
            if w_start < e and w_end > s:
                return spk
        
        # Second: check if word is in a small gap before next speaker (likely timing misalignment)
        for i, (s, e, spk) in enumerate(turns):
            if w_end <= s:  # Word is before this speaker
                gap = s - w_end
                if gap <= gap_threshold:
                    # Word is very close to this speaker's turn - likely belongs to them
                    return spk
                break
        
        # Third: most recent speaker (turn ends closest before word starts)
        before = [turn for turn in turns if turn[1] <= w_start]
        if before:
            return max(before, key=lambda x: x[1])[2]
        
        # Fourth: nearest speaker overall
        mid = (w_start + w_end) / 2
        return min(turns, key=lambda x: abs(mid - (x[0] + x[1]) / 2))[2]

    result = []
    for w in word_segments:
        w_start = w.get("start", 0.0)
        w_end = w.get("end", 0.0)
        result.append({**w, "speaker": speaker_at(w_start, w_end)})
    return result


def group_into_utterances(word_speaker_list: List[dict], diarization=None) -> List[dict]:
    """
    Merge consecutive words from the same speaker into
    utterance blocks: {speaker, start, end, text}.
    
    If diarization is provided, snap utterance boundaries to the
    actual speaker turn boundaries from diarization (authoritative source).
    """
    if not word_speaker_list:
        return []

    # Extract diarization turns if provided
    diar_turns = []
    if diarization is not None:
        diar_turns = [
            (seg.start, seg.end, spk)
            for seg, _, spk in diarization.itertracks(yield_label=True)
        ]
        diar_turns.sort(key=lambda x: x[0])

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
    
    # Snap utterance boundaries to diarization turns if available
    if diar_turns:
        for utt in utterances:
            spk = utt["speaker"]
            # Find all diarization turns for this speaker
            spk_turns = [t for t in diar_turns if t[2] == spk]
            if spk_turns:
                # Snap to earliest start and latest end of this speaker's turns
                # that overlap with this utterance's current boundaries
                overlapping = [
                    t for t in spk_turns
                    if not (t[1] < utt["start"] or t[0] > utt["end"])
                ]
                if overlapping:
                    utt["start"] = min(t[0] for t in overlapping)
                    utt["end"] = max(t[1] for t in overlapping)
    
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

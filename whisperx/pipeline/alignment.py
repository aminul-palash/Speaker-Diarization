"""
pipeline/alignment.py
WhisperX forced alignment — produces word-level timestamps
which are required for accurate speaker assignment.
"""

import gc
import torch
import whisperx


def align_words(
    segments: list,
    prepared_wav: str,
    language: str,
    device: str,
) -> list:
    """
    Align ASR segments to audio using wav2vec2 for the given language.
    Returns a flat list of word dicts: {word, start, end, score}.

    Falls back to evenly-distributed segment timestamps if the
    wav2vec2 model for 'bn' is unavailable.
    """
    print(f"[ALIGN] Loading wav2vec2 alignment model for '{language}'...")

    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
        audio   = whisperx.load_audio(prepared_wav)
        aligned = whisperx.align(
            segments,
            align_model,
            align_meta,
            audio,
            device,
            return_char_alignments=False,
        )
        word_segments = aligned.get("word_segments", [])
        print(f"   ✅ {len(word_segments)} words aligned")

        # Preview
        for w in word_segments[:5]:
            print(f"      {w.get('word',''):25s}  "
                  f"[{w.get('start',0):.2f}s → {w.get('end',0):.2f}s]  "
                  f"score={w.get('score',0):.2f}")

        del align_model
        gc.collect()
        torch.cuda.empty_cache()
        return word_segments

    except Exception as e:
        print(f"   ⚠️  Alignment failed: {e}")
        print("   Falling back to evenly-distributed segment timestamps.")
        return _segment_level_fallback(segments)


def _segment_level_fallback(segments: list) -> list:
    """
    Build a word list from segment-level timestamps by distributing
    words evenly across each segment's duration.
    Not as accurate as wav2vec2 alignment but still usable for
    diarization assignment.
    """
    word_segments = []
    for seg in segments:
        words = seg["text"].split()
        if not words:
            continue
        dur = (seg["end"] - seg["start"]) / max(len(words), 1)
        for j, word in enumerate(words):
            word_segments.append({
                "word":  word,
                "start": seg["start"] + j * dur,
                "end":   seg["start"] + (j + 1) * dur,
                "score": 0.0,
            })
    print(f"   Fallback word list: {len(word_segments)} entries")
    return word_segments

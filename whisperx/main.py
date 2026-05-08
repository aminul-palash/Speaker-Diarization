"""
main.py
Entry point for the Bangla clinical ASR + diarization pipeline.

Usage:
    # Basic run (uses config/config.py values)
    python main.py

    # Override audio file
    python main.py --audio path/to/audio.wav

    # Override number of speakers
    python main.py --audio audio.wav --min-speakers 2 --max-speakers 3

    # Use a fine-tuned Whisper checkpoint
    python main.py --whisper-model output/whisper_clinical_bn

    # Use a fine-tuned Pyannote segmentation checkpoint
    python main.py --seg-ckpt output/pyannote_clinical/segmentation_clinical.ckpt

    # Use LLM role classifier instead of heuristic
    python main.py --openai-key sk-...
"""

import argparse
import os
import shutil
from pathlib import Path

import torch

import config.config as cfg
from pipeline.asr         import load_asr_model, transcribe, free_asr_model
from pipeline.alignment   import align_words
from pipeline.diarization import load_diarization_pipeline, run_diarization
from utils.audio          import convert_to_wav_mono_16k
from utils.speaker        import (
    assign_speakers_to_words,
    group_into_utterances,
    speaker_stats,
    heuristic_role_map,
)
from utils.role_classifier import llm_role_map
from utils.writers         import write_txt, write_srt, write_json


# ── Argument parser ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bangla Clinical ASR + Diarization Pipeline"
    )
    parser.add_argument(
        "--audio", default=cfg.AUDIO_PATH,
        help="Path to audio file (wav, mp3, m4a, flac …)"
    )
    parser.add_argument(
        "--whisper-model", default=cfg.WHISPER_MODEL,
        help="Whisper model ID or path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--min-speakers", type=int, default=cfg.MIN_SPEAKERS,
        help="Minimum number of speakers for diarization"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=cfg.MAX_SPEAKERS,
        help="Maximum number of speakers for diarization"
    )
    parser.add_argument(
        "--seg-ckpt", default=None,
        help="Path to fine-tuned Pyannote segmentation checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--openai-key", default=None,
        help="OpenAI API key for LLM-based role classification"
    )
    parser.add_argument(
        "--output-dir", default=cfg.OUTPUT_DIR,
        help="Directory for output files"
    )
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    base = Path(args.audio).stem

    # ── Stage 1: Audio preprocessing ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 1 — Audio Preprocessing")
    print("=" * 55)
    prepared_wav = os.path.join(temp_dir, f"{base}_16k.wav")
    convert_to_wav_mono_16k(args.audio, prepared_wav)

    # ── Stage 2: ASR transcription ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 2 — Bangla ASR Transcription")
    print("=" * 55)
    asr_model, asr_backend = load_asr_model(
        args.whisper_model, cfg.DEVICE, cfg.COMPUTE_TYPE, cfg.LANGUAGE
    )
    result = transcribe(
        prepared_wav, asr_model, asr_backend,
        cfg.DEVICE, cfg.LANGUAGE, cfg.BATCH_SIZE
    )
    detected_lang = result.get("language", cfg.LANGUAGE)
    free_asr_model(asr_model, asr_backend)

    # ── Stage 3: Forced alignment ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 3 — Forced Alignment (word timestamps)")
    print("=" * 55)
    word_segments = align_words(
        result["segments"], prepared_wav, cfg.LANGUAGE, cfg.DEVICE
    )

    # ── Stage 4: Diarization ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 4 — Speaker Diarization")
    print("=" * 55)
    diar_pipeline = load_diarization_pipeline(
        cfg.DIARIZATION_MODEL, cfg.HF_TOKEN, cfg.DEVICE
    )

    # Optionally swap in a fine-tuned segmentation model
    if args.seg_ckpt and os.path.isfile(args.seg_ckpt):
        print(f"[DIAR] Swapping in fine-tuned segmentation: {args.seg_ckpt}")
        from pyannote.audio import Model as PyannoteModel
        ft_seg = PyannoteModel.load_from_checkpoint(args.seg_ckpt)
        ft_seg.to(cfg.DEVICE)
        diar_pipeline._segmentation.model = ft_seg
        print("   ✅ Fine-tuned segmentation model loaded")

    diarization, speakers_found = run_diarization(
        diar_pipeline, prepared_wav,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # ── Stage 5: Speaker assignment & grouping ────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 5 — Speaker Assignment & Utterance Grouping")
    print("=" * 55)
    words_with_speakers = assign_speakers_to_words(word_segments, diarization)
    utterances          = group_into_utterances(words_with_speakers)
    stats               = speaker_stats(utterances)

    print(f"   {len(utterances)} utterances across {len(stats)} speakers")
    print("\n   Speaker statistics:")
    for spk, s in sorted(stats.items(), key=lambda x: -x[1]["words"]):
        print(f"      {spk:<15s}  words={s['words']:>5d}   "
              f"time={s['seconds']:>6.1f}s")

    # Preview
    print("\n   First 5 utterances:")
    for u in utterances[:5]:
        print(f"      [{u['speaker']} @ {u['start']:.1f}s]  {u['text'][:80]}")

    # ── Stage 6: Speaker role assignment ──────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 6 — Speaker Role Assignment")
    print("=" * 55)

    role_map = {}

    if args.openai_key:
        print("   Using LLM-based role classifier...")
        role_map = llm_role_map(utterances, api_key=args.openai_key)

    if not role_map:
        print("   Using heuristic role assignment (most words → doctor)...")
        role_map = heuristic_role_map(utterances)

    print("\n   Role map:")
    for spk, role in role_map.items():
        s = stats.get(spk, {})
        print(f"      {spk:<15s} → {role:<12s}  "
              f"(words={s.get('words', 0)}, "
              f"time={s.get('seconds', 0):.1f}s)")

    # ── Stage 7: Save outputs ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 7 — Saving Outputs")
    print("=" * 55)

    meta = {
        "audio_file":   args.audio,
        "language":     detected_lang,
        "asr_model":    args.whisper_model,
        "diar_model":   cfg.DIARIZATION_MODEL,
        "speakers":     speakers_found,
        "role_map":     role_map,
        "n_utterances": len(utterances),
    }

    txt_path  = os.path.join(args.output_dir, f"{base}_transcript.txt")
    srt_path  = os.path.join(args.output_dir, f"{base}_transcript.srt")
    json_path = os.path.join(args.output_dir, f"{base}_transcript.json")

    write_txt (utterances, txt_path,  role_map)
    write_srt (utterances, srt_path,  role_map)
    write_json(utterances, json_path, role_map, meta)

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n   Temp files removed.")
    print(f"\n✅ Pipeline complete. Outputs in: {args.output_dir}")
    print(f"   {Path(txt_path).name}")
    print(f"   {Path(srt_path).name}")
    print(f"   {Path(json_path).name}")

    return txt_path, srt_path, json_path


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)

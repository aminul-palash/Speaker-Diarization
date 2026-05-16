"""
pipeline/asr.py
Bangla ASR using WhisperX with a fine-tuned HuggingFace model.

Load strategy:
  1. Try WhisperX native loader (faster-whisper CTranslate2 backend).
  2. If that fails, fall back to HuggingFace transformers.
"""

import gc
import torch
import whisperx
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_asr_model(model_id: str, device: str, compute_type: str, language: str):
    """
    Load ASR model. Returns (model_or_tuple, backend_str).
    backend_str is either "whisperx" or "hf".
    """
    print(f"[ASR] Loading model: {model_id}")

    try:
        model = whisperx.load_model(
            model_id,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        print("   ✅ Loaded via WhisperX (faster-whisper backend)")
        return model, "whisperx"

    except Exception as e:
        print(f"   ℹ️  WhisperX native failed ({type(e).__name__}), "
              f"switching to HF transformers...")

    processor = WhisperProcessor.from_pretrained(model_id)
    hf_model  = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    print("   ✅ Loaded via HuggingFace transformers")
    return (processor, hf_model), "hf"


def transcribe(
    prepared_wav: str,
    model,
    backend: str,
    device: str,
    language: str,
    batch_size: int = 8,
) -> dict:
    """
    Transcribe audio and return a whisperx-compatible result dict:
      {"segments": [{start, end, text}, ...], "language": "bn"}
    """
    print(f"[ASR] Transcribing ({backend} backend)...")

    if backend == "whisperx":
        audio  = whisperx.load_audio(prepared_wav)
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
        )

    else:  # hf transformers — chunked 30-second inference
        processor, hf_model = model
        result = _transcribe_hf(prepared_wav, processor, hf_model, device)

    n = len(result["segments"])
    lang = result.get("language", language)
    print(f"   ✅ Done — {n} segment(s) | detected language: {lang}")

    # Preview
    for seg in result["segments"][:3]:
        print(f"      [{seg['start']:.1f}s → {seg['end']:.1f}s]  "
              f"{seg['text'][:80]}")

    return result


def free_asr_model(model, backend: str):
    """Release GPU memory after transcription."""
    if backend == "whisperx":
        del model
    else:
        processor, hf_model = model
        del processor, hf_model
    gc.collect()
    torch.cuda.empty_cache()
    print("[ASR] GPU memory freed.")


# ── HF transformers chunked inference ────────────────────────────────────

def _transcribe_hf(
    audio_path: str,
    processor,
    model,
    device: str,
    chunk_s: float = 30.0,
) -> dict:
    import librosa
    from transformers import GenerationConfig
    
    # Update generation config with language/task mappings if missing
    if not hasattr(model.generation_config, 'task_to_id'):
        try:
            # Load reference config from official Whisper model
            ref_config = GenerationConfig.from_pretrained("openai/whisper-large-v2")
            model.generation_config.task_to_id = ref_config.task_to_id
            model.generation_config.lang_to_id = ref_config.lang_to_id
        except Exception:
            pass  # Fallback: model might still work without these
    
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    chunk     = int(chunk_s * 16000)
    segments = []

    for start in range(0, len(audio), chunk):
        piece  = audio[start: start + chunk]
        inputs = processor(piece, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        feats  = inputs.input_features.to(device)
        attention_mask = inputs.get("attention_mask", None)
        if model.dtype == torch.float16:
            feats = feats.half()
        gen_kwargs = dict(
            inputs=feats,
            task="transcribe",
            language="bengali",
            max_new_tokens=440,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask.to(device)
        with torch.no_grad():
            ids = model.generate(**gen_kwargs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        t0   = start / 16000
        t1   = t0 + len(piece) / 16000
        segments.append({"start": t0, "end": t1, "text": text})

    return {"segments": segments, "language": "bn"}

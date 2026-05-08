"""
pipeline/diarization.py
Pyannote speaker diarization — language-agnostic, works on
speaker embeddings so it handles Bangla clinical audio natively.
"""

import torch
from pyannote.audio import Pipeline


def load_diarization_pipeline(model_id: str, hf_token: str, device: str):
    """Load and return a Pyannote diarization pipeline."""
    print(f"[DIAR] Loading pipeline: {model_id}")
    pipeline = Pipeline.from_pretrained(
        model_id,
        token=hf_token,
    ).to(torch.device(device))
    print("   ✅ Pyannote diarization ready")
    return pipeline


def run_diarization(
    pipeline,
    audio_path: str,
    min_speakers: int = None,
    max_speakers: int = None,
):
    """
    Run diarization and return the Pyannote Annotation object.
    min_speakers / max_speakers constrain the number of speakers;
    pass None to let Pyannote auto-detect.
    """
    import librosa
    from pyannote.core import Annotation
    
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    print(f"[DIAR] Running on {audio_path}  "
          f"(min={min_speakers}, max={max_speakers})")

    # Load audio as in-memory dict to avoid torchcodec issues
    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
    audio_dict = {
        "waveform": torch.from_numpy(waveform).unsqueeze(0).float(),
        "sample_rate": sr,
    }
    
    result = pipeline(audio_dict, **kwargs)
    
    # Handle newer pyannote output format (DiarizeOutput dataclass)
    # Extract speaker_diarization annotation from result
    if hasattr(result, 'speaker_diarization'):
        diarization = result.speaker_diarization
    elif hasattr(result, 'segmentation'):
        diarization = result.segmentation
    else:
        diarization = result

    # Collect speakers from annotation
    speakers = sorted({
        spk for _, _, spk in diarization.itertracks(yield_label=True)
    })
    print(f"   ✅ {len(speakers)} speaker(s) found: {speakers}")

    # Preview first 6 turns
    for i, (turn, _, spk) in enumerate(diarization.itertracks(yield_label=True)):
        if i >= 6:
            break
        print(f"      {spk:<15s}  [{turn.start:.2f}s → {turn.end:.2f}s]")

    return diarization, speakers

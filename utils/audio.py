"""
utils/audio.py
Audio preprocessing utilities.
"""

import torchaudio


def convert_to_wav_mono_16k(audio_path: str, out_path: str) -> str:
    """
    Convert any audio file (mp3, m4a, flac, wav …) to
    16 kHz mono WAV — the format required by Whisper and Pyannote.
    """
    waveform, sr = torchaudio.load(audio_path)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=16000
        )(waveform)

    torchaudio.save(out_path, waveform, 16000)

    duration = waveform.shape[-1] / 16000
    print(f"   Audio ready: {duration:.1f}s  →  {out_path}")
    return out_path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import torch
import tempfile
import os
import numpy as np
import librosa

from transformers import WhisperForConditionalGeneration, WhisperProcessor

app = FastAPI()

# ── Load model once at startup (VERY IMPORTANT) ─────────────────────────────
MODEL_PATH = "/home/technonext/Downloads/Speaker-Diarization-main/ASR/model/whisper-small-librispeech"
FALLBACK_MODEL = "openai/whisper-small"
LANGUAGE = "english"
TASK = "transcribe"
SAMPLING_RATE = 16000


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
try:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
except:
    processor = WhisperProcessor.from_pretrained(FALLBACK_MODEL, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(FALLBACK_MODEL)

model = model.to(device)
model.eval()
print("Model loaded ✔")


# ── Helpers ────────────────────────────────────────────────────────────────
def load_audio(path: str):
    audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
    return audio


def transcribe_chunk(audio: np.ndarray):
    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def parse_rttm_file(content: str):
    segments = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append({
                "start": start,
                "end": start + dur,
                "speaker": speaker
            })
    return segments


def extract_segment(audio, start, end):
    return audio[int(start * SAMPLING_RATE): int(end * SAMPLING_RATE)]


# ── API Endpoint ────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    rttm: UploadFile = File(...)
):
    # Save temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(await audio.read())
        audio_path = tmp_audio.name

    rttm_text = (await rttm.read()).decode("utf-8")

    try:
        waveform = load_audio(audio_path)
        segments = parse_rttm_file(rttm_text)

        results = []

        for seg in segments:
            audio_seg = extract_segment(waveform, seg["start"], seg["end"])

            if len(audio_seg) == 0:
                text = "[silence]"
            else:
                text = transcribe_chunk(audio_seg)

            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text
            })

        return JSONResponse({
            "segments": results
        })

    finally:
        os.remove(audio_path)
from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import os
import tempfile
from pathlib import Path
from pyannote.audio import Pipeline
from safetensors.torch import load_file
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Speaker Diarization API")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env")

DEVICE = torch.device("cpu")  # change to "cuda" if available

# ─────────────────────────────────────────────
# LOAD PIPELINE ONCE (IMPORTANT)
# ─────────────────────────────────────────────
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Load custom segmentation model
checkpoint_path = Path("/home/technonext/Downloads/Speaker-Diarization-main/segmentation/model")

seg_model = pipeline._segmentation.model

if (checkpoint_path / "model.safetensors").exists():
    state_dict = load_file(checkpoint_path / "model.safetensors")
elif (checkpoint_path / "pytorch_model.bin").exists():
    state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=DEVICE)
else:
    raise FileNotFoundError("No model weights found")

seg_model.load_state_dict(state_dict, strict=False)
seg_model = seg_model.to(DEVICE)

pipeline._segmentation.model = seg_model
pipeline = pipeline.to(DEVICE)

print("✔ Diarization pipeline ready")


# ─────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────
def run_diarization(audio_path: str):
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "duration": float(turn.end - turn.start),
            "speaker": speaker
        })

    return segments


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "title": "Speaker Diarization API",
        "version": "1.0",
        "endpoints": {
            "POST /diarize": "Upload an audio file for speaker diarization"
        },
        "usage": "Send a POST request with audio file to /diarize"
    }


@app.get("/diarize")
async def diarize_info():
    return {
        "message": "Use POST method with audio file",
        "example": "curl -X POST -F 'audio=@audio.wav' http://0.0.0.0:8001/diarize"
    }


@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    try:
        segments = run_diarization(audio_path)

        return {
            "num_segments": len(segments),
            "num_speakers": len(set(s["speaker"] for s in segments)),
            "segments": segments
        }

    finally:
        os.remove(audio_path)
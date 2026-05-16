import os
import torch

# ─────────────────────────────────────────────────────────
#  REQUIRED — change before running
# ─────────────────────────────────────────────────────────
AUDIO_PATH = "data/consultation.wav"   # path to your audio file
HF_TOKEN   = "hugging_face_key"      # huggingface read token

# ─────────────────────────────────────────────────────────
#  ASR MODEL  (uncomment one)
# ─────────────────────────────────────────────────────────
# Best overall — trained on large Bengali.AI corpus
WHISPER_MODEL = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"

# Regional dialects
# WHISPER_MODEL = "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"

# Lighter option
# WHISPER_MODEL = "asif00/whisper-bangla"

# Vanilla multilingual fallback
# WHISPER_MODEL = "large-v3"

# ─────────────────────────────────────────────────────────
#  DIARIZATION
# ─────────────────────────────────────────────────────────
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
MIN_SPEAKERS      = 2    # at minimum doctor + patient
MAX_SPEAKERS      = 5    # doctor + patient + attendants

# ─────────────────────────────────────────────────────────
#  PIPELINE SETTINGS
# ─────────────────────────────────────────────────────────
LANGUAGE     = "bn"
BATCH_SIZE   = 8          # reduce to 4 if CUDA OOM
COMPUTE_TYPE = "float16"  # use "int8" on low-VRAM GPUs
OUTPUT_DIR   = "output"

# ─────────────────────────────────────────────────────────
#  FINE-TUNE SETTINGS (only needed if you fine-tune)
# ─────────────────────────────────────────────────────────
FT_WHISPER_BASE   = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
FT_WHISPER_OUTPUT = "output/whisper_clinical_bn"
FT_TRAIN_CSV      = "data/train.csv"   # columns: audio_path, transcript
FT_EVAL_CSV       = "data/eval.csv"
FT_EPOCHS         = 3
FT_BATCH_SIZE     = 8
FT_LR             = 1e-5

FT_PYANNOTE_DATA_DIR = "data/clinical_diarize"   # see finetune/finetune_pyannote.py
FT_PYANNOTE_OUTPUT   = "output/pyannote_clinical"
FT_PYANNOTE_EPOCHS   = 10

# ─────────────────────────────────────────────────────────
#  AUTO-RESOLVED
# ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

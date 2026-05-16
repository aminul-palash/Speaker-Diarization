# Bangla Clinical ASR + Speaker Diarization Pipeline

End-to-end pipeline for transcribing doctor–patient consultations
in Bengali with multi-speaker identification.

---

## Project Structure

```
bangla_asr_pipeline/
│
├── config/
│   └── config.py             # all settings — edit this first
│
├── pipeline/
│   ├── asr.py                # Bangla Whisper transcription
│   ├── alignment.py          # WhisperX word-level alignment
│   └── diarization.py        # Pyannote speaker diarization
│
├── utils/
│   ├── audio.py              # audio conversion (→ 16 kHz mono WAV)
│   ├── speaker.py            # word→speaker mapping, utterance grouping
│   └── writers.py            # TXT / SRT / JSON output
│
├── finetune/
│   ├── finetune_whisper.py   # fine-tune Whisper on clinical transcripts
│   └── finetune_pyannote.py  # fine-tune Pyannote segmentation on RTTM data
│
├── data/                     # put your audio files here
├── output/                   # generated transcripts appear here
│
├── main.py                   # pipeline entry point
├── requirements.txt
└── README.md
```

---

## Setup

Python version used: `3.10`

```bash
pip install -r requirements.txt
```

### HuggingFace token
You need a HuggingFace **read** token and must accept the licence for:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Set your token in `config/config.py`:
```python
HF_TOKEN = "hf_your_token_here"
```

---

## Usage

### 1. Basic run
```bash
python main.py --audio data/consultation.wav
```

### 2. Override number of speakers
```bash
python main.py --audio data/consultation.wav --min-speakers 2 --max-speakers 3
```

### 3. Use fine-tuned models
```bash
python main.py \
  --audio data/consultation.wav \
  --whisper-model output/whisper_clinical_bn \
  --seg-ckpt output/pyannote_clinical/segmentation_clinical.ckpt
```

---

## Fine-tuning

### Whisper — needs (audio, transcript) CSV pairs
```
data/train.csv  →  columns: audio_path, transcript
data/eval.csv   →  same format
```
```bash
python -m finetune.finetune_whisper
```
Saved to `output/whisper_clinical_bn/`.

### Pyannote segmentation — needs RTTM annotations
```
data/clinical_diarize/
    audio/    rec001.wav  rec002.wav ...
    rttm/     rec001.rttm rec002.rttm ...
    train.txt             (one file stem per line)
    val.txt
```
RTTM line format:
```
SPEAKER rec001 1 0.50 2.30 <NA> <NA> doctor <NA> <NA>
```
```bash
python -m finetune.finetune_pyannote
```
Saved to `output/pyannote_clinical/segmentation_clinical.ckpt`.

---

## Output Files

| File | Description |
|------|-------------|
| `*_transcript.txt`  | Plain text with speaker labels |
| `*_transcript.srt`  | SRT subtitles with timestamps |
| `*_transcript.json` | Machine-readable with metadata |

---

## Models Used

| Stage | Model |
|-------|-------|
| ASR | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` |
| Alignment | WhisperX wav2vec2 (`bn`) |
| Diarization | `pyannote/speaker-diarization-3.1` |

## Why no Demucs / vocal separation?
Clinical audio contains speech + ambient noise (AC, fans) — not music.
Whisper's built-in VAD handles this cleanly. Demucs adds overhead
without improving transcription quality on this type of data.

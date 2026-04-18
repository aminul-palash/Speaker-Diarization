# Speaker Diarization with Fine-tuned Pyannote Segmentation

## Overview
Fine-tuning pyannote/segmentation-3.0 on a filtered VoxConverse subset for
multi-speaker diarization. Answers "who spoke when" in audio recordings.

---

## Requirements

- Python 3.10
- PyTorch 2.1.0
- pyannote.audio 3.1.1
- datasets 2.19.0
- diarizers (from GitHub)
- HuggingFace account with accepted terms for:
  - pyannote/segmentation-3.0
  - pyannote/speaker-diarization-3.1

## Setup

Create and activate virtual environment:
```bash
python3.10 -m venv myenv
source myenv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset

- Source: diarizers-community/voxconverse (YouTube-based, English)
- Selection criteria: samples with >= 3 unique speakers
- Final subset: 30 samples split into train/validation/test (70/15/15)
- Pushed to Hub: aminulpalash506/voxconverse-30-subset
- Columns: audio, timestamps_start, timestamps_end, speakers

VoxConverse was chosen over AMI because file durations (5-10 min vs 20-60 min)
are manageable within Kaggle's 30GB RAM without crashing during preprocessing.

---

## Preprocessing

Handled internally by diarizers train_segmentation.py:
- Resamples audio to 16kHz mono
- Segments into overlapping chunks using pyannote Preprocess()
- Extracts speaker labels per chunk from timestamps

No manual preprocessing required when using raw dataset with correct columns.

---

## Training

Platform: Kaggle (P100 GPU, 30GB RAM)
Base model: pyannote/segmentation-3.0

```bash
python train_segmentation.py \
    --dataset_name=aminulpalash506/voxconverse-30-subset \
    --train_split_name=train \
    --eval_split_name=validation \
    --model_name_or_path=pyannote/segmentation-3.0 \
    --output_dir=./finetuned-segmentation-voxconverse \
    --do_train --do_eval \
    --learning_rate=1e-3 \
    --num_train_epochs=3 \
    --lr_scheduler_type=cosine \
    --per_device_train_batch_size=4 \
    --eval_strategy=steps --save_strategy=steps \
    --save_steps=10 --eval_steps=10 \
    --preprocessing_num_workers=0 \
    --dataloader_num_workers=0
```

Training time: ~6 minutes on P100

---

## Results

| Metric           | Value  |
|------------------|--------|
| Loss             | 0.3543 |
| DER              | 8.85%  |
| False Alarm      | 2.51%  |
| Missed Detection | 2.17%  |
| Confusion        | 4.16%  |

Best DER observed during training: 7.90% at step 460 (epoch 1.15).
Model converged around epoch 1.5 with minimal improvement after.

---

## Inference

Load the fine-tuned segmentation model from local checkpoint:

```python
import torch
from pyannote.audio import Pipeline
from safetensors.torch import load_file
from pathlib import Path

device = torch.device("cpu")

# Load base pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# Load local fine-tuned weights
checkpoint_path = Path("./checkpoint-1200")
seg_model = pipeline._segmentation.model
state_dict = load_file(checkpoint_path / "model.safetensors")
seg_model.load_state_dict(state_dict, strict=False)

# Run on audio file
pipeline._segmentation.model = seg_model.to(device)
pipeline = pipeline.to(device)
diarization = pipeline("audio.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f} {turn.end:.2f} {speaker}")
```

Or use the provided script:
```bash
source myenv/bin/activate
python inference.py
```

---

## Checkpoint Structure

The checkpoint directory contains:
- `model.safetensors` - Fine-tuned model weights
- `config.json` - Model configuration
- `pytorch_model.bin` - Alternative weights format (if available)

## Notes

- AMI dataset was initially attempted but caused OOM crashes due to long files
- CallHome was considered but VoxConverse has naturally higher speaker count
- 30-sample subset is intentionally small for assignment scope
- Larger dataset and more epochs would further reduce DER

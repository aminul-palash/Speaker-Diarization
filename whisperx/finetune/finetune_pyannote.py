"""
finetune/finetune_pyannote.py

Fine-tune the Pyannote segmentation model on your clinical audio.

Requirements:
    pip install pyannote.database pytorch-lightning

Data folder structure:
    data/clinical_diarize/
        audio/          *.wav  (16 kHz mono)
        rttm/           *.rttm (one per audio, same stem)
        train.txt       (one file stem per line)
        val.txt

RTTM format (one speaker segment per line):
    SPEAKER rec001 1 0.50 2.30 <NA> <NA> doctor <NA> <NA>
    SPEAKER rec001 1 3.10 4.80 <NA> <NA> patient <NA> <NA>

Usage:
    python -m finetune.finetune_pyannote

After training:
    Use the checkpoint path printed at the end in main.py
    by passing --seg-ckpt <path> or editing config/config.py.
"""

import os
import sys
import torch
import pytorch_lightning as pl

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    HF_TOKEN,
    FT_PYANNOTE_DATA_DIR,
    FT_PYANNOTE_OUTPUT,
    FT_PYANNOTE_EPOCHS,
    DEVICE,
)

from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation
from pyannote.database import registry


def write_db_config(data_dir: str) -> str:
    """Write the pyannote.database YAML config for your clinical data."""
    cfg = f"""
Protocols:
  ClinicalBn:
    SpeakerDiarization:
      Consultation:
        train:
          uri: {data_dir}/train.txt
          annotation: {data_dir}/rttm/{{uri}}.rttm
          audio: {data_dir}/audio/{{uri}}.wav
        development:
          uri: {data_dir}/val.txt
          annotation: {data_dir}/rttm/{{uri}}.rttm
          audio: {data_dir}/audio/{{uri}}.wav
"""
    path = os.path.join(data_dir, "database.yml")
    with open(path, "w") as f:
        f.write(cfg)
    print(f"   Database config written → {path}")
    return path


def main():
    os.makedirs(FT_PYANNOTE_OUTPUT, exist_ok=True)

    cfg_path = write_db_config(FT_PYANNOTE_DATA_DIR)
    registry.load_database(cfg_path)
    protocol = registry.get_protocol(
        "ClinicalBn.SpeakerDiarization.Consultation"
    )

    print(f"Loading base segmentation model from pyannote/segmentation-3.0 ...")
    seg_model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=HF_TOKEN,
    )

    seg_model.task = Segmentation(
        protocol,
        duration=2.0,        # 2-second chunks during training
        max_num_speakers=4,  # doctor + patient + 2 attendants max
    )

    trainer = pl.Trainer(
        max_epochs=FT_PYANNOTE_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=FT_PYANNOTE_OUTPUT,
    )

    print("Starting Pyannote segmentation fine-tuning...")
    trainer.fit(seg_model)

    ckpt = os.path.join(FT_PYANNOTE_OUTPUT, "segmentation_clinical.ckpt")
    trainer.save_checkpoint(ckpt)

    print(f"\n✅ Fine-tuned segmentation saved → {ckpt}")
    print("\nTo use in pipeline, pass --seg-ckpt to main.py:")
    print(f"   python main.py --seg-ckpt {ckpt}")


if __name__ == "__main__":
    main()

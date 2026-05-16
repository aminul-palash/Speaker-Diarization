"""
finetune/finetune_whisper.py

Fine-tune a Bangla Whisper model on your own clinical data.

Requirements:
    pip install datasets evaluate jiwer

Data format — two CSV files (train.csv and eval.csv):
    audio_path,transcript
    data/audio/rec001.wav,আপনার কি সমস্যা হচ্ছে?
    data/audio/rec002.wav,জ্বর আছে তিন দিন ধরে।

Usage:
    python -m finetune.finetune_whisper

After training:
    Set WHISPER_MODEL = FT_WHISPER_OUTPUT in config/config.py
    and run main.py normally.
"""

import os
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict, Union

from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    FT_WHISPER_BASE, FT_WHISPER_OUTPUT,
    FT_TRAIN_CSV, FT_EVAL_CSV,
    FT_EPOCHS, FT_BATCH_SIZE, FT_LR,
    DEVICE,
)


# ── Dataset ───────────────────────────────────────────────────────────────

def load_csv_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path).rename(
        columns={"audio_path": "audio", "transcript": "sentence"}
    )
    return Dataset.from_pandas(df).cast_column(
        "audio", Audio(sampling_rate=16000)
    )


# ── Data collator ─────────────────────────────────────────────────────────

@dataclass
class SpeechCollator:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        inputs["labels"] = labels
        return inputs


# ── Training ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(FT_WHISPER_OUTPUT, exist_ok=True)

    print(f"Loading processor from: {FT_WHISPER_BASE}")
    processor = WhisperProcessor.from_pretrained(FT_WHISPER_BASE)
    processor.tokenizer.set_prefix_tokens(
        language="bengali", task="transcribe"
    )

    def prepare(batch):
        a = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            a["array"], sampling_rate=a["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("Loading datasets...")
    train_ds = load_csv_dataset(FT_TRAIN_CSV).map(
        prepare, remove_columns=["audio", "sentence"]
    )
    eval_ds = load_csv_dataset(FT_EVAL_CSV).map(
        prepare, remove_columns=["audio", "sentence"]
    )

    print(f"Loading model: {FT_WHISPER_BASE}")
    model = WhisperForConditionalGeneration.from_pretrained(FT_WHISPER_BASE)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        lab_ids  = pred.label_ids
        lab_ids[lab_ids == -100] = processor.tokenizer.pad_token_id
        p_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        l_str = processor.tokenizer.batch_decode(
            lab_ids, skip_special_tokens=True
        )
        return {"wer": round(
            wer_metric.compute(predictions=p_str, references=l_str), 4
        )}

    args = Seq2SeqTrainingArguments(
        output_dir=FT_WHISPER_OUTPUT,
        per_device_train_batch_size=FT_BATCH_SIZE,
        per_device_eval_batch_size=FT_BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=FT_LR,
        warmup_steps=50,
        num_train_epochs=FT_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SpeechCollator(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting Whisper fine-tuning...")
    trainer.train()
    trainer.save_model(FT_WHISPER_OUTPUT)
    processor.save_pretrained(FT_WHISPER_OUTPUT)

    print(f"\n✅ Fine-tuned Whisper saved → {FT_WHISPER_OUTPUT}")
    print(f"   Set WHISPER_MODEL = '{FT_WHISPER_OUTPUT}' in config/config.py")


if __name__ == "__main__":
    main()

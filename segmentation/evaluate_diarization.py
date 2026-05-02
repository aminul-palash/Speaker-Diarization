import os
import torch
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# ============================================================
# CONFIG
# ============================================================

HF_DATASET = "aminulpalash506/voxconverse-30-subset"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

OUTPUT_CSV = "evaluation_results.csv"
COLLAR = 0.25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD PIPELINE
# ============================================================

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
pipeline = pipeline.to(device)

# ============================================================
# HELPER: Convert HF annotation → pyannote Annotation
# ============================================================

def hf_to_annotation(hf_segments):
    ann = Annotation()
    for seg in hf_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        spk = str(seg["speaker"])
        ann[Segment(start, end)] = spk
    return ann


# ============================================================
# OVERLAP FUNCTION
# ============================================================

def get_overlap(seg1, seg2):
    start = max(seg1.start, seg2.start)
    end = min(seg1.end, seg2.end)
    return max(0.0, end - start)


# ============================================================
# SEGMENT TABLE
# ============================================================

def generate_segment_table(gt_ann, pred_ann, file_id):
    rows = []

    for gt_seg, _, gt_spk in gt_ann.itertracks(yield_label=True):
        for pred_seg, _, pred_spk in pred_ann.itertracks(yield_label=True):

            overlap = get_overlap(gt_seg, pred_seg)

            if overlap > 0:
                rows.append({
                    "file": file_id,
                    "gt_start": round(gt_seg.start, 2),
                    "gt_end": round(gt_seg.end, 2),
                    "gt_speaker": gt_spk,
                    "pred_start": round(pred_seg.start, 2),
                    "pred_end": round(pred_seg.end, 2),
                    "pred_speaker": pred_spk,
                    "overlap_duration": round(overlap, 2),
                    "speaker_match": int(gt_spk == pred_spk)
                })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    dataset = load_dataset(HF_DATASET, split="train")

    metric = DiarizationErrorRate(collar=COLLAR)
    all_rows = []

    print(f"Loaded {len(dataset)} samples\n")

    for i, sample in enumerate(dataset):
        file_id = f"sample_{i}"

        # ----------------------------
        # Audio
        # ----------------------------
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"]).unsqueeze(0)
        sample_rate = audio["sampling_rate"]

        # ----------------------------
        # Ground Truth
        # ----------------------------
        gt_segments = sample["segments"]  # expected format
        gt_ann = hf_to_annotation(gt_segments)

        # ----------------------------
        # Prediction
        # ----------------------------
        pred_ann = pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
        })

        # ----------------------------
        # DER
        # ----------------------------
        der = metric(gt_ann, pred_ann)
        components = metric.compute_components(gt_ann, pred_ann)

        miss = components["missed detection"]
        fa = components["false alarm"]
        conf = components["confusion"]

        print(f"{file_id}: DER={der:.3f} | Miss={miss:.3f} | FA={fa:.3f} | Conf={conf:.3f}")

        # ----------------------------
        # Segment table
        # ----------------------------
        rows = generate_segment_table(gt_ann, pred_ann, file_id)

        for r in rows:
            r["DER"] = round(der, 4)
            r["Miss"] = round(miss, 4)
            r["FalseAlarm"] = round(fa, 4)
            r["Confusion"] = round(conf, 4)

        all_rows.extend(rows)

    # ============================================================
    # SAVE CSV
    # ============================================================

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved CSV: {OUTPUT_CSV}")

    # ============================================================
    # GLOBAL DER
    # ============================================================

    overall_der = abs(metric)
    print("\n==============================")
    print(f"GLOBAL DER: {overall_der:.4f}")
    print("==============================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
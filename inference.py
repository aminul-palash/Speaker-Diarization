import torch
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline
from safetensors.torch import load_file
import json
from pathlib import Path

device = torch.device("cpu")

# Load pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
)

# Load fine-tuned segmentation model from local checkpoint
checkpoint_path = Path("/home/aminul/Documents/speaker/Speaker-Diarization/checkpoint-1200")

# Get the segmentation model from the pipeline
seg_model = pipeline._segmentation.model

# Load model weights from checkpoint
# Try loading safetensors format first, then pytorch format
if (checkpoint_path / "model.safetensors").exists():
    state_dict = load_file(checkpoint_path / "model.safetensors")
    print(f"Loaded model.safetensors - {len(state_dict)} parameters")
elif (checkpoint_path / "pytorch_model.bin").exists():
    state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=device)
    print(f"Loaded pytorch_model.bin - {len(state_dict)} parameters")
else:
    raise FileNotFoundError("No model weights found in checkpoint folder")

# Load weights into the model
seg_model.load_state_dict(state_dict, strict=False)
seg_model = seg_model.to(device)

# Replace and finalize
pipeline._segmentation.model = seg_model
pipeline = pipeline.to(device)
print("Pipeline ready with local fine-tuned model\n")

# ============================================================
# Inference on any audio file
# ============================================================
def run_inference(audio_path):
    print(f"\nRunning inference on: {audio_path}")
    diarization = pipeline(audio_path)

    segments = []
    print(f"\n{'START':>8} {'END':>8} {'DUR':>6}  SPEAKER")
    print("-" * 40)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        segments.append({
            "start": round(turn.start, 2),
            "end"  : round(turn.end, 2),
            "dur"  : round(dur, 2),
            "speaker": speaker
        })
        print(f"{turn.start:>8.2f} {turn.end:>8.2f} {dur:>6.2f}  {speaker}")

    print(f"\nTotal speakers detected: {len(set(s['speaker'] for s in segments))}")
    return segments

def format_by_time_slots(segments, time_window=5.0):
    """Format diarization output grouped by time slots with speakers.
    
    Args:
        segments: List of segment dictionaries from run_inference
        time_window: Duration of each time slot in seconds (default 5s)
    
    Returns:
        Formatted string organized by time slots
    """
    if not segments:
        return "No segments found"
    
    # Group segments by time slot
    max_time = max(s["end"] for s in segments)
    num_slots = int(np.ceil(max_time / time_window))
    
    output = []
    output.append(f"\nDIARIZATION BY TIME SLOTS (window: {time_window}s)\n")
    output.append("=" * 60)
    
    for slot_idx in range(num_slots):
        slot_start = slot_idx * time_window
        slot_end = (slot_idx + 1) * time_window
        
        output.append(f"\nTime Slot {slot_idx + 1}: {slot_start:.1f}s - {slot_end:.1f}s")
        output.append("-" * 60)
        
        # Find segments in this time slot
        speakers_in_slot = {}
        for seg in segments:
            # Check if segment overlaps with this time slot
            if seg["start"] < slot_end and seg["end"] > slot_start:
                speaker = seg["speaker"]
                if speaker not in speakers_in_slot:
                    speakers_in_slot[speaker] = []
                speakers_in_slot[speaker].append(seg)
        
        if not speakers_in_slot:
            output.append("(No speech)")
        else:
            # Sort speakers by name
            for speaker in sorted(speakers_in_slot.keys()):
                segs = speakers_in_slot[speaker]
                output.append(f"\n  {speaker}:")
                for seg in segs:
                    # Clip segment to time slot boundaries
                    seg_start = max(seg["start"], slot_start)
                    seg_end = min(seg["end"], slot_end)
                    dur = seg_end - seg_start
                    output.append(f"    {seg_start:.2f}s - {seg_end:.2f}s ({dur:.2f}s)")
    
    return "\n".join(output)

def save_rttm(segments, audio_path, out_path="output.rttm"):
    name = Path(audio_path).stem
    with open(out_path, "w") as f:
        for s in segments:
            f.write(f"SPEAKER {name} 1 {s['start']:.2f} {s['dur']:.2f} "
                    f"<NA> <NA> {s['speaker']} <NA> <NA>\n")
    print(f"Saved RTTM to {out_path}")

if __name__ == "__main__":
    audio_file = "/home/aminul/Documents/speaker/Speaker-Diarization/sample_0.wav"  # Change this to your audio file path
    segments = run_inference(audio_file)
    
    # Print time slot format
    formatted_output = format_by_time_slots(segments, time_window=5.0)
    print(formatted_output)
    
    # Save to text file
    with open("diarization_time_slots.txt", "w") as f:
        f.write(formatted_output)
    print("\nSaved time slot format to diarization_time_slots.txt")
    
    # Save RTTM format
    save_rttm(segments, audio_file, out_path="diarization_output.rttm")
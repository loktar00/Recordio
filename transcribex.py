import whisperx
import gc
import json
import os

device = "cuda"
audio_file = "10-29-leadership meeting.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs

# Modified code to generate JSON file with specific information
# Get the base name of the audio file without extension
base_name = os.path.splitext(os.path.basename(audio_file))[0]
output_file = f"{base_name}_transcription.json"

# Create a new dictionary with only the desired information
simplified_result = {
    "segments": [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker": segment.get("speaker", "Unknown")  # Use "Unknown" if speaker is not present
        }
        for segment in result["segments"]
    ]
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(simplified_result, f, ensure_ascii=False, indent=2)

print(f"Simplified transcription result saved to {output_file}")
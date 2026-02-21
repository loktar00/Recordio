import os
import shutil
import whisperx
import gc
import json
from pathlib import Path

def setup_directories():
    """Create necessary directories if they don't exist."""
    Path("./raw recordings").mkdir(exist_ok=True)
    Path("./transcripts").mkdir(exist_ok=True)

def is_audio_file(filename):
    """Check if file is an audio file based on extension."""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac'}
    return os.path.splitext(filename)[1].lower() in audio_extensions

def process_audio_file(audio_file, device="cuda", batch_size=16, compute_type="float16"):
    """Process a single audio file and return the transcription result."""
    try:
        # 1. Load model and transcribe
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # Clear GPU memory
        del model
        gc.collect()

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Clear GPU memory
        del model_a
        gc.collect()

        # 3. Assign speaker labels

        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="INSERT TOKEN HERE INVALIDATED", device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return result

    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return None

def save_transcription(result, audio_file):
    """Save transcription result to JSON file."""
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = f"./transcripts/{base_name}_transcription.json"

    simplified_result = {
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment.get("speaker", "Unknown")
            }
            for segment in result["segments"]
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(simplified_result, f, ensure_ascii=False, indent=2)

def move_audio_file(audio_file):
    """Move processed audio file to raw recordings directory."""
    destination = os.path.join("./raw recordings", os.path.basename(audio_file))
    shutil.move(audio_file, destination)

def main():
    # setup_directories()

    # Get all audio files in the current directory
    audio_files = [f for f in os.listdir('.') if os.path.isfile(f) and is_audio_file(f)]

    if not audio_files:
        print("No audio files found in the current directory.")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file}")

        # Process the audio file
        result = process_audio_file(audio_file)

        if result:
            # Save transcription
            save_transcription(result, audio_file)
            print(f"Transcription saved for {audio_file}")

            # Move audio file
            move_audio_file(audio_file)
            print(f"Moved {audio_file} to raw recordings directory")
        else:
            print(f"Failed to process {audio_file}")

if __name__ == "__main__":
    main()

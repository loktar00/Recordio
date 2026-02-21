# recordio

Batch audio transcription pipeline using [WhisperX](https://github.com/m-bain/whisperX). Drop audio files into the project root, run `batch_transcribe.py`, and get speaker-diarized transcripts as JSON. Processed audio files are moved out of the way automatically.

## Requirements

- Python 3.8+
- CUDA-capable GPU (runs on `cuda` with `float16` by default)
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- A HuggingFace auth token for the speaker diarization model (already hardcoded in the script)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
recordio/
├── batch_transcribe.py      # Main script - processes all audio files in the root
├── requirements.txt         # Python dependencies
├── raw recordings/          # Processed audio files get moved here automatically
└── transcripts/             # JSON transcription output lands here
```

## How It Works

1. **Drop audio files** (`.wav`, `.mp3`, `.m4a`, `.flac`, `.aac`) into the project root directory.

2. **Run the batch transcriber:**
   ```bash
   python batch_transcribe.py
   ```

3. **For each audio file**, the script:
   - Loads the WhisperX `large-v2` model and transcribes the audio
   - Aligns the transcript segments for accurate timestamps
   - Runs speaker diarization to label who said what
   - Saves a simplified JSON transcript to `transcripts/`
   - Moves the original audio file to `raw recordings/`

4. **Output format** (`transcripts/<filename>_transcription.json`):
   ```json
   {
     "segments": [
       {
         "start": 0.718,
         "end": 1.779,
         "text": "Some transcribed text.",
         "speaker": "SPEAKER_03"
       }
     ]
   }
   ```

   Each segment includes a start/end timestamp (in seconds), the transcribed text, and a speaker label.

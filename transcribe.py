import whisper
import sys
import subprocess
import os
import warnings
import io
import contextlib

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

@contextlib.contextmanager
def suppress_stdout_stderr():
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def main():
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Load the Whisper model while suppressing all output
    with suppress_stdout_stderr():
        # Load the Whisper model (you can choose between "tiny", "base", "small", "medium", "large")
        model = whisper.load_model("base")

    # Check if ffmpeg is installed
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not in the system PATH.")
        print("Please install ffmpeg and make sure it's accessible from the command line.")
        print("You can download ffmpeg from: https://ffmpeg.org/download.html")
        sys.exit(1)

    # Check if an audio file was provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)

    # Load the audio file passed as a command-line argument
    audio_file = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(audio_file):
        print(f"Error: The audio file '{audio_file}' does not exist.")
        sys.exit(1)

    try:
        # Perform the transcription
        result = model.transcribe(audio_file)

        # Print the transcribed text
        print(result['text'])
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        print("Please make sure the audio file is in a supported format.")
        sys.exit(1)

if __name__ == "__main__":
    main()
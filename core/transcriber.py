"""Speech to text with a locally loaded Whisper model."""

from pathlib import Path


def transcribe_audio(audio_path: Path, model) -> dict:
    return model.transcribe(str(audio_path), fp16=False)


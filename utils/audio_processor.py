"""Extract mono 16 kHz WAV audio for Whisper."""

import subprocess
from pathlib import Path


def extract_audio(video_path: Path, audio_path: Path) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
             "-ar", "16000", str(audio_path)],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required. Install it and add it to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not extract audio: {exc.stderr[-800:]}") from exc


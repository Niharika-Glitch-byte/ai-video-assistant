# AI Video Assistant

A Streamlit app that turns video speech into a transcript, a short extractive summary and key points. Ask questions about the video and see the transcript passages used to generate the answer.

## Features

- Upload an MP4, MOV, MKV or WebM video, or paste a YouTube link.
- Extract audio with FFmpeg and transcribe speech locally with OpenAI Whisper (`base`, CPU).
- Pick summary sentences and key points using TF-IDF similarity.
- Retrieve relevant transcript chunks with TF-IDF and answer with `google/flan-t5-small`.
- Keep question and answer history during the Streamlit session.

## Run locally

1. Install Python 3.10+ and FFmpeg. Verify that `ffmpeg -version` works in your terminal.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. Install packages and start the app:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

The first transcription and Q&A can take longer while the models download. Video and transcript processing run on your machine. YouTube downloading requires a network connection and may depend on video availability.

## How it works

`app.py` handles the interface. `utils/audio_processor.py` extracts 16 kHz mono audio. `core/transcriber.py` calls Whisper. `core/vector_store.py` splits the transcript and ranks chunks by TF-IDF similarity. The app selects representative summary sentences and prompts FLAN-T5 with retrieved context for question answering.

## Limitations

Summaries are extractive. Generated answers may be inaccurate, so check the displayed source text. Performance depends on video length, speech clarity and CPU speed. Only process videos you have permission to use.

## Resume description

**AI Video Assistant** — Built a Streamlit video analysis app with Whisper transcription, TF-IDF based summarization and transcript retrieval, and FLAN-T5 question answering for uploaded and YouTube videos.

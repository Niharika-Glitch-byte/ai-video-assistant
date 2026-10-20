# AI Video Assistant

A Streamlit app that turns video speech into a transcript, a short extractive summary and key points. Ask questions about the video and see the transcript passages used to generate the answer.

## Features

- Upload an MP4, MOV, MKV or WebM video, or paste a YouTube link. For YouTube, the app tries public captions first, then audio; you can also paste a transcript manually.
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

The first transcription and Q&A can take longer while the models download. Processing runs where the app is hosted (your machine when run locally, or the Streamlit server when deployed). YouTube downloading requires a network connection and may depend on video availability.

On hosted platforms, YouTube may reject both caption and audio requests from the cloud server. In that case, open Show transcript on YouTube and paste the text into the app, or upload a video file. This app does not require or collect YouTube cookies.

## How it works

`app.py` handles the interface. `utils/audio_processor.py` extracts 16 kHz mono audio. `core/transcriber.py` calls Whisper. `core/vector_store.py` splits the transcript and ranks chunks by TF-IDF similarity. The app selects representative summary sentences and prompts FLAN-T5 with retrieved context for question answering.

## Limitations

Summaries are extractive. Generated answers may be inaccurate, so check the displayed source text. Performance depends on video length, speech clarity and CPU speed. Only process videos you have permission to use.

## Resume description

**AI Video Assistant** — Built a Streamlit video analysis app with Whisper transcription, TF-IDF based summarization and transcript retrieval, and FLAN-T5 question answering for uploaded and YouTube videos.

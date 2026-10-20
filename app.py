from pathlib import Path
from tempfile import TemporaryDirectory
from utils.audio_processor import extract_audio
from core.transcriber import transcribe_audio
from core.vector_store import create_vector_store, search_chunks
from core.youtube_captions import fetch_caption_text, video_id_from_url
import streamlit as st
import whisper
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Video Assistant", page_icon="🎬")
st.title("🎬 AI Video Assistant")
st.write("Upload a video to transcribe, summarize, and ask questions about it.")


@st.cache_resource
def load_whisper():
    return whisper.load_model("base", device="cpu")


def save_transcript(text, source):
    st.session_state["transcript"] = text
    collection, embedding_model = create_vector_store(text)
    st.session_state["video_store"] = (source, collection, embedding_model)
    st.session_state["chat_history"] = []


youtube_url = st.text_input("Or paste a YouTube video URL")

if st.button("Analyze YouTube video") and youtube_url.strip():
    from yt_dlp import YoutubeDL

    st.session_state.pop("youtube_bytes", None)
    st.session_state.pop("youtube_transcript_url", None)
    st.session_state.pop("video_store", None)
    try:
        video_id_from_url(youtube_url)
    except ValueError as error:
        st.error(str(error))
    else:
        caption_text = None
        with st.spinner("Checking for public captions..."):
            try:
                caption_text = fetch_caption_text(youtube_url)
            except Exception:
                pass
        if caption_text:
            save_transcript(caption_text, youtube_url)
            st.session_state["youtube_transcript_url"] = youtube_url
            st.session_state.pop("youtube_bytes", None)
            st.success("YouTube captions are ready for analysis.")
        else:
            try:
                with st.spinner("Captions unavailable. Trying video audio..."):
                    with TemporaryDirectory() as folder:
                        options = {
                            "format": "bestaudio[ext=m4a]/bestaudio",
                            "outtmpl": str(Path(folder) / "%(id)s.%(ext)s"),
                            "noplaylist": True,
                            "quiet": True,
                        }
                        with YoutubeDL(options) as ydl:
                            info = ydl.extract_info(youtube_url.strip(), download=True)
                            downloaded = Path(info["requested_downloads"][0]["filepath"])
                        st.session_state["youtube_bytes"] = downloaded.read_bytes()
                        st.session_state["youtube_name"] = downloaded.name
                st.session_state.pop("youtube_transcript_url", None)
                st.success("YouTube audio is ready. Click Transcribe video below.")
            except Exception:
                st.warning(
                    "YouTube did not provide captions or blocked this cloud server. "
                    "Paste the transcript below, or upload a video file."
                )

if youtube_url.strip():
    with st.expander("Have the YouTube transcript? Paste it here"):
        st.caption("On YouTube, open the video's description, choose Show transcript, then copy its text here.")
        pasted_text = st.text_area("Video transcript", height=160)
        if st.button("Analyze pasted transcript"):
            try:
                video_id_from_url(youtube_url)
                if not pasted_text.strip():
                    st.warning("Paste a transcript first.")
                else:
                    save_transcript(pasted_text.strip(), youtube_url)
                    st.session_state["youtube_transcript_url"] = youtube_url
                    st.session_state.pop("youtube_bytes", None)
                    st.success("Transcript is ready for analysis.")
            except ValueError as error:
                st.error(str(error))

video = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "webm"])
if video is None and "youtube_bytes" in st.session_state:
    from io import BytesIO
    video = BytesIO(st.session_state["youtube_bytes"])
    video.name = st.session_state["youtube_name"]
if video:
    st.success(f"Selected video: {video.name}")
    if video.name == st.session_state.get("youtube_name"):
        st.video(youtube_url)
    else:
        st.video(video)
    if st.button("Transcribe video"):
        with TemporaryDirectory() as folder:
            video_path = Path(folder) / Path(video.name).name
            audio_path = Path(folder) / "audio.wav"
            video_path.write_bytes(video.getbuffer())

            extract_audio(video_path, audio_path)
            st.audio(audio_path.read_bytes(), format="audio/wav")

            with st.spinner("Transcribing speech..."):
                model = load_whisper()
                result = transcribe_audio(audio_path, model)
                save_transcript(result["text"], video.name)
                st.session_state.pop("youtube_transcript_url", None)
if st.session_state.get("youtube_transcript_url"):
    st.video(st.session_state["youtube_transcript_url"])
if "video_store" in st.session_state:
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    collection = st.session_state["video_store"][1]
    chunks = collection.get(include=["documents"])["documents"]
    transcript = st.session_state.get("transcript") or " ".join(chunks)
    st.subheader("Transcript")
    st.write(transcript)
    summary_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()
    ]
    if summary_sentences and any(ch.isalnum() for ch in transcript):
        summary_vectors = TfidfVectorizer().fit_transform(summary_sentences)
        summary_scores = cosine_similarity(summary_vectors).mean(axis=1)
        count = max(1, min(3, (len(summary_sentences) + 2) // 3))
        selected = sorted(summary_scores.argsort()[-count:])
        st.subheader("Short summary")
        st.write(" ".join(summary_sentences[i] for i in selected))
    sentences = list(dict.fromkeys(
        s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript)
        if len(s.split()) >= 8
    ))

    if len(sentences) >= 2:
        try:
            vectors = TfidfVectorizer(stop_words="english").fit_transform(sentences)
            scores = cosine_similarity(vectors, np.asarray(vectors.mean(axis=0))).ravel()
            chosen = sorted(scores.argsort()[-5:])
            st.subheader("Key points")
            st.markdown("\n".join(f"- {sentences[i]}" for i in chosen))
        except ValueError:
            pass
if "video_store" in st.session_state:
    st.subheader("Ask a question about the video")
    question = st.text_input("Your question")
    if st.button("Find answer") and question.strip():
        _, collection, embedding_model = st.session_state["video_store"]
        matches = search_chunks(collection, embedding_model, question)
        if not matches:
            st.info("No relevant passage was found in the video transcript.")
        else:
          with st.spinner("Generating an answer..."):
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            if "qa_model" not in st.session_state:
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
                st.session_state["qa_model"] = (tokenizer, qa_model)

            tokenizer, qa_model = st.session_state["qa_model"]
            context = "\n\n".join(matches)[:1800]
            prompt = (
    "Use the context to answer the question in one short sentence.\n"
    f"Context: {context}\nQuestion: {question}\nAnswer:"
)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            output = qa_model.generate(**inputs, max_new_tokens=60)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
          st.subheader("Answer")
          st.write(answer)
          st.session_state.setdefault("chat_history", []).append((question, answer))
          with st.expander("Source passage from the transcript"):
              st.write(context)
if st.session_state.get("chat_history"):
    st.subheader("Previous questions and answers")
    for old_question, old_answer in st.session_state["chat_history"]:
        st.markdown(f"**Question:** {old_question}\n\n**Answer:** {old_answer}")

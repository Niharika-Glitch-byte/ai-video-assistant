from pathlib import Path
from tempfile import TemporaryDirectory
from utils.audio_processor import extract_audio
from core.transcriber import transcribe_audio
from core.vector_store import create_vector_store, search_chunks
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


youtube_url = st.text_input("Or paste a YouTube video URL")

if st.button("Load YouTube video") and youtube_url.strip():
    from yt_dlp import YoutubeDL

    try:
        with st.spinner("Downloading YouTube audio..."):
            with TemporaryDirectory() as folder:
                options = {
                   "format": "bestaudio[ext=m4a]/bestaudio",
                    "outtmpl": str(Path(folder) / "%(id)s.%(ext)s"),
                    "noplaylist": True,
                }
                with YoutubeDL(options) as ydl:
                    info = ydl.extract_info(youtube_url.strip(), download=True)
                    downloaded = Path(info["requested_downloads"][0]["filepath"])
                st.session_state["youtube_bytes"] = downloaded.read_bytes()
                st.session_state["youtube_name"] = downloaded.name
        st.success("YouTube audio is ready!")
    except Exception as error:
        if "Sign in to confirm you’re not a bot" in str(error) or "Sign in to confirm you're not a bot" in str(error):
            st.warning(
                "YouTube blocked downloads from this cloud server. "
                "If you have the video file, use the upload option below."
            )
        else:
            st.error(f"Could not load the YouTube video: {error}")

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
                st.session_state["transcript"] = result["text"]
                collection, embedding_model = create_vector_store(result["text"])
                st.session_state["video_store"] = (video.name, collection, embedding_model)
            st.subheader("Transcript")
            st.write(result["text"])
            import re
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", result["text"])
                if s.strip()
            ]

            if sentences and any(ch.isalnum() for ch in result["text"]):
                vectors = TfidfVectorizer().fit_transform(sentences)
                scores = cosine_similarity(vectors).mean(axis=1)
                count = max(1, min(3, (len(sentences) + 2) // 3))
                selected = sorted(scores.argsort()[-count:])

                st.subheader("Short summary")
                st.write(" ".join(sentences[i] for i in selected))
if "video_store" in st.session_state:
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    collection = st.session_state["video_store"][1]
    chunks = collection.get(include=["documents"])["documents"]
    transcript = st.session_state.get("transcript") or " ".join(chunks)
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

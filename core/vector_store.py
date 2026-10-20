"""Lightweight in-memory TF-IDF retrieval over transcript chunks."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TranscriptCollection:
    def __init__(self, documents):
        self.documents = documents

    def get(self, include=None):
        return {"documents": self.documents}


def _chunks(text, max_words=100, overlap=20):
    words = text.split()
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]


def create_vector_store(transcript):
    documents = _chunks(transcript.strip())
    if not documents:
        return TranscriptCollection([]), None
    model = TfidfVectorizer(stop_words="english")
    try:
        model.fit(documents)
    except ValueError:
        # A short transcript can contain only stop words.
        model = TfidfVectorizer()
        model.fit(documents)
    return TranscriptCollection(documents), model


def search_chunks(collection, embedding_model, question, top_k=3):
    if not collection.documents or embedding_model is None:
        return []
    matrix = embedding_model.transform(collection.documents)
    query = embedding_model.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [collection.documents[i] for i in ranked if scores[i] > 0]


"""Fetch publicly available YouTube captions without downloading video audio."""

from urllib.parse import parse_qs, urlparse


def video_id_from_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Enter a valid YouTube URL.")
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/live/"):
            video_id = parsed.path.split("/")[2]
        else:
            video_id = ""
    elif host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        video_id = ""
    if len(video_id) != 11 or not all(ch.isalnum() or ch in "-_" for ch in video_id):
        raise ValueError("Enter a valid YouTube video URL.")
    return video_id


def fetch_caption_text(url: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    snippets = YouTubeTranscriptApi().fetch(video_id_from_url(url))
    # Captions are typically only a few words each. Group them into useful
    # passages for sentence-based summary and retrieval.
    lines = [item.text.strip().replace("\n", " ") for item in snippets if item.text.strip()]
    return " ".join(
        " ".join(lines[i:i + 8]).strip() + "."
        for i in range(0, len(lines), 8)
    ).strip()

import re
from urllib.parse import urlparse, parse_qs

from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a YouTube URL."""

    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ValueError(
        "Invalid YouTube URL. Please provide a valid YouTube video URL."
    )

def load_youtube(url: str):
    """
    Load transcript content from a YouTube video.

    Args:
        url: YouTube video URL.

    Returns:
        List containing a LangChain Document object.
    """

    video_id = extract_video_id(url)

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)

    selected_transcript = next(iter(transcript_list))

    language = selected_transcript.language
    language_code = selected_transcript.language_code

    fetched_transcript = selected_transcript.fetch()

    text = " ".join(
        snippet.text for snippet in fetched_transcript
    )

    document = Document(
        page_content=text,
        metadata={
            "source": url,
            "video_id": video_id,
            "type": "youtube",
            "language": language,
            "language_code": language_code
        }
    )

    return [document]
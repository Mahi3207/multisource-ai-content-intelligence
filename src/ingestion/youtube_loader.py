import logging
import os
import re
import tempfile
from functools import lru_cache

from langchain_core.documents import Document

from src.config import TRANSCRIPT_WINDOW_SECONDS, WHISPER_MODEL_SIZE
from src.exception import ContentLoadError


logger = logging.getLogger(__name__)

VIDEO_ID_PATTERNS = [
    r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})",
]


def extract_video_id(url: str) -> str:
    for pattern in VIDEO_ID_PATTERNS:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ContentLoadError(
        "Invalid YouTube URL. Please provide a valid YouTube video URL."
    )


def _format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    return f"{minutes}:{seconds:02d}"


def build_documents(
    snippets,
    url,
    video_id,
    extra_metadata,
    window_seconds=None,
):
    window_seconds = window_seconds or TRANSCRIPT_WINDOW_SECONDS

    documents = []
    window_start = None
    window_texts = []

    def flush():
        text = " ".join(window_texts).strip()

        if not text:
            return

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "video_id": video_id,
                    "type": "youtube",
                    "start_seconds": int(window_start or 0),
                    "timestamp": _format_timestamp(window_start or 0),
                    **extra_metadata,
                },
            )
        )

    for start, text in snippets:
        text = text.strip()

        if not text:
            continue

        if window_start is None:
            window_start = start

        if start - window_start >= window_seconds and window_texts:
            flush()
            window_start = start
            window_texts = []

        window_texts.append(text)

    if window_texts:
        flush()

    return documents


def _fetch_transcript_snippets(video_id: str):
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript_list = YouTubeTranscriptApi().list(video_id)

    try:
        # Try English first, then use the first available transcript.
        transcript = transcript_list.find_transcript(
            ["en", "en-US", "en-GB"]
        )
    except Exception:
        transcript = next(iter(transcript_list))

    fetched = transcript.fetch()
    snippets = [(snippet.start, snippet.text) for snippet in fetched]

    return snippets, transcript.language, transcript.language_code


@lru_cache(maxsize=1)
def _get_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )


def _download_audio(video_id: str, target_dir: str) -> str:
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(target_dir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=True,
        )

        return downloader.prepare_filename(info)


def _transcribe_with_whisper(video_id: str):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        audio_path = _download_audio(video_id, temp_dir)

        model = _get_whisper_model()
        segments, info = model.transcribe(
            audio_path,
            vad_filter=True,
        )

        # Transcription happens when the generator is consumed.
        snippets = [
            (segment.start, segment.text)
            for segment in segments
        ]

    return snippets, info.language


def load_youtube(url: str):
    video_id = extract_video_id(url)

    # Try YouTube's transcript first.
    try:
        snippets, language, language_code = _fetch_transcript_snippets(
            video_id
        )
        method = "transcript"

    except Exception as e:
        logger.info(
            "Transcript unavailable for %s. Using Whisper.",
            video_id,
        )
        snippets = []

    # Use Whisper when a transcript isn't available.
    if not snippets:
        try:
            snippets, language_code = _transcribe_with_whisper(video_id)
            language = language_code
            method = "whisper"

        except Exception as e:
            logger.exception(
                "Whisper transcription failed for %s",
                video_id,
            )
            raise ContentLoadError(
                "Could not get the transcript or audio from this video."
            ) from e

    documents = build_documents(
        snippets,
        url,
        video_id,
        {
            "language": language,
            "language_code": language_code,
            "transcription_method": method,
        },
    )

    if not documents:
        raise ContentLoadError("No speech could be found in this video.")

    return documents


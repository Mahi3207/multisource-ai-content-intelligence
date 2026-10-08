"""Central configuration. Values can be overridden with environment variables."""

import os

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

RETRIEVAL_K = 4


ANALYSIS_BATCH_CHARS = 12_000

ANALYSIS_MERGE_CHARS = 16_000

MAX_ANALYSIS_CHARS = int(os.getenv("MAX_ANALYSIS_CHARS", "400000"))


WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

TRANSCRIPT_WINDOW_SECONDS = 120


TOO_LARGE_MESSAGE = (
    "⚠️ This file is too large for reliable AI analysis.\n\n"
    "Please upload a shorter document so the system can provide more "
    "accurate summaries, key points, and insights."
)
NOT_FOUND_MESSAGE = "I could not find the answer in the provided content."

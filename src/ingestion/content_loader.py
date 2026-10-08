import os

import validators

from src.exception import ContentLoadError
from src.ingestion.docx_loader import load_docx
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.web_loader import load_website
from src.ingestion.youtube_loader import load_youtube


SUPPORTED_SOURCE_TYPES = {"pdf", "docx", "youtube", "website"}


def detect_source_type(source: str) -> str:
    source = source.lower().strip()

    if source.startswith(("http://", "https://")):
        if "youtube.com/" in source or "youtu.be/" in source:
            return "youtube"

        if not validators.url(source):
            raise ValueError("The URL does not look valid.")

        return "website"

    if source.endswith(".pdf"):
        return "pdf"

    if source.endswith(".docx"):
        return "docx"

    raise ValueError(
        "Could not determine the source type. "
        "Supported sources are PDF, DOCX, YouTube, and website URLs."
    )


def load_content(
    source: str,
    source_type: str = None,
    display_name: str = None,
):
    source = source.strip()

    if source_type is None:
        source_type = detect_source_type(source)

    source_type = source_type.lower().strip()

    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported source type: {source_type}. "
            "Supported types: pdf, docx, youtube, website."
        )

    if source_type in {"pdf", "docx"} and not os.path.isfile(source):
        raise ContentLoadError("The file could not be found.")

    loaders = {
        "pdf": load_pdf,
        "docx": load_docx,
        "youtube": load_youtube,
        "website": load_website,
    }

    documents = loaders[source_type](source)

    if display_name:
        for document in documents:
            document.metadata["source"] = display_name

    return documents


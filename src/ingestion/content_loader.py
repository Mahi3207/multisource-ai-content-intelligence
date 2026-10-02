from src.ingestion.pdf_loader import load_pdf
from src.ingestion.docx_loader import load_docx
from src.ingestion.youtube_loader import load_youtube
from src.ingestion.web_loader import load_website


SUPPORTED_SOURCE_TYPES = {
    "pdf",
    "docx",
    "youtube",
    "website"
}


def detect_source_type(source: str) -> str:
    """
    Detect the type of content source from a file path or URL.

    Args:
        source: File path or URL.

    Returns:
        Detected source type.
    """

    source_lower = source.lower().strip()

    # YouTube URL
    if (
        "youtube.com/" in source_lower
        or "youtu.be/" in source_lower
    ):
        return "youtube"

    # PDF file
    if source_lower.endswith(".pdf"):
        return "pdf"

    # DOCX file
    if source_lower.endswith(".docx"):
        return "docx"

    # Other HTTP/HTTPS URLs are treated as websites
    if (
        source_lower.startswith("http://")
        or source_lower.startswith("https://")
    ):
        return "website"

    raise ValueError(
        "Could not determine the source type. "
        "Supported sources: PDF, DOCX, YouTube, and website URLs."
    )


def load_content(source: str, source_type: str = None):
    """
    Load content using the appropriate source-specific loader.

    If source_type is not provided, it is automatically detected.

    Args:
        source: File path or URL.
        source_type: Optional source type.

    Returns:
        List of LangChain Document objects.
    """

    if source_type is None:
        source_type = detect_source_type(source)

    source_type = source_type.lower().strip()

    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported source type: {source_type}. "
            "Supported types: pdf, docx, youtube, website."
        )

    if source_type == "pdf":
        return load_pdf(source)

    if source_type == "docx":
        return load_docx(source)

    if source_type == "youtube":
        return load_youtube(source)

    if source_type == "website":
        return load_website(source)
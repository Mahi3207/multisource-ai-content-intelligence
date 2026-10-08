from langchain_community.document_loaders import PyPDFLoader

from src.exception import ContentLoadError


def load_pdf(file_path: str):
    try:
        documents = PyPDFLoader(file_path).load()

    except Exception as e:
        raise ContentLoadError(
            "Could not read the PDF. It may be corrupted or password protected."
        ) from e

    for document in documents:
        document.metadata["type"] = "pdf"

    # Make sure the PDF actually contains some readable text.
    if not any(document.page_content.strip() for document in documents):
        raise ContentLoadError(
            "No readable text was found in the PDF. "
            "It may be a scanned document."
        )

    return documents


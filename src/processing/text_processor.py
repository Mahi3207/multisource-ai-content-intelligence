import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents):
    cleaned_documents = []

    for document in documents:
        cleaned_text = clean_text(document.page_content)

        if cleaned_text:
            cleaned_documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata=dict(document.metadata),
                )
            )

    return cleaned_documents


def split_documents(documents, doc_id: str = None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(clean_documents(documents))

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

        if doc_id is not None:
            chunk.metadata["doc_id"] = doc_id

    return chunks


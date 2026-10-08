from dataclasses import dataclass, field
from typing import Any, List

from src.config import RETRIEVAL_K
from src.exceptions import ContentLoadError
from src.ingestion.content_loader import detect_source_type, load_content
from src.intelligence.content_analyzer import analyze_content, check_size
from src.processing.text_processor import clean_documents, split_documents
from src.retrieval.retriever import create_retriever
from src.retrieval.vector_store import (
    create_vector_store,
    delete_vector_store,
    new_document_id,
)


@dataclass
class ProcessedContent:
    doc_id: str
    name: str
    source_type: str
    total_chars: int
    chunks: List[Any] = field(repr=False)
    vector_store: Any = field(repr=False)
    retriever: Any = field(repr=False)
    analysis: str = ""


def process_source(
    source: str,
    embedding_model,
    llm,
    source_type: str = None,
    display_name: str = None,
    progress=None,
) -> ProcessedContent:
    source_type = source_type or detect_source_type(source)

    documents = load_content(
        source,
        source_type,
        display_name=display_name,
    )
    documents = clean_documents(documents)

    if not documents:
        raise ContentLoadError("No text could be extracted from this source.")

    # Check the size before doing any embedding work.
    total_chars = check_size(documents)

    doc_id = new_document_id()
    chunks = split_documents(documents, doc_id=doc_id)

    if not chunks:
        raise ContentLoadError("No usable text chunks were generated.")

    vector_store = create_vector_store(
        chunks,
        embedding_model,
        doc_id,
    )

    try:
        analysis = analyze_content(
            documents,
            llm,
            progress=progress,
        )
    except Exception:
        delete_vector_store(vector_store)
        raise

    retriever = create_retriever(
        vector_store,
        doc_id,
        k=RETRIEVAL_K,
    )

    return ProcessedContent(
        doc_id=doc_id,
        name=display_name or source,
        source_type=source_type,
        total_chars=total_chars,
        chunks=chunks,
        vector_store=vector_store,
        retriever=retriever,
        analysis=analysis,
    )


def release_content(content: ProcessedContent) -> None:
    if content is not None:
        delete_vector_store(content.vector_store)
import uuid

import chromadb
from langchain_chroma import Chroma


def new_document_id() -> str:
    return uuid.uuid4().hex


def _simple_metadata(metadata: dict) -> dict:
    # Chroma supports only simple metadata values.
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def create_vector_store(chunks, embedding_model, doc_id: str):
    if not chunks:
        raise ValueError("Cannot create a vector store without chunks.")

    for chunk in chunks:
        chunk.metadata = _simple_metadata(chunk.metadata)
        chunk.metadata["doc_id"] = doc_id

    vector_store = Chroma(
        collection_name=f"doc_{doc_id}",
        embedding_function=embedding_model,
        client=chromadb.EphemeralClient(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    vector_store.add_documents(chunks)

    return vector_store


def delete_vector_store(vector_store) -> None:
    try:
        vector_store.delete_collection()
    except Exception:
        pass
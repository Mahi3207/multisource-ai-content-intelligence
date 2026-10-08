from langchain_chroma import Chroma


COLLECTION_NAME = "content_intelligence"


def create_vector_store(documents, embedding_model):
    """
    Create an in-memory Chroma vector store.

    A fresh vector store is created every time content
    is processed, preventing different documents from
    being mixed together.

    Args:
        documents: List of LangChain Document objects.
        embedding_model: Embedding model used for vector generation.

    Returns:
        Chroma vector store.
    """

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME
    )

    return vector_store
from langchain_chroma import Chroma


VECTOR_STORE_PATH = "data/vector_store"
COLLECTION_NAME = "content_intelligence"


def create_vector_store(documents, embedding_model):
    """
    Create a persistent Chroma vector store from LangChain documents.

    Args:
        documents: List of LangChain Document objects.
        embedding_model: Embedding model used for vector generation.

    Returns:
        Persistent Chroma vector store.
    """

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_PATH
    )

    return vector_store
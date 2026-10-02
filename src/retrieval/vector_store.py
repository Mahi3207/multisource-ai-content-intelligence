from langchain_chroma import Chroma


def create_vector_store(documents, embedding_model):
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name="content_intelligence"
    )

    return vector_store
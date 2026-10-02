from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


def generate_embeddings(texts, embedding_model):
    
    if not texts:
        return []

    return embedding_model.embed_documents(texts)
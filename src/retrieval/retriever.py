def create_retriever(vector_store, doc_id: str, k: int = 4):
    return vector_store.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"doc_id": doc_id},
        }
    )
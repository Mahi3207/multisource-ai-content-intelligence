from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    

    text = text.strip()

    text = " ".join(text.split())

    return text


def split_documents(documents):
    
    cleaned_documents = []

    for document in documents:
        cleaned_text = clean_text(document.page_content)

        if cleaned_text:
            document.page_content = cleaned_text
            cleaned_documents.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(cleaned_documents)

    return chunks
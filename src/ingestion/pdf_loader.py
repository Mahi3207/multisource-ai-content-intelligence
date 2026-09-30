from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """
    Load text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents
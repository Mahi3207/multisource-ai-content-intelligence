from docx import Document
from langchain_core.documents import Document as LangChainDocument


def load_docx(file_path: str):
    """
    Load text from a DOCX file.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        List of LangChain Document objects.
    """

    doc = Document(file_path)

    documents = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            documents.append(
                LangChainDocument(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "type": "docx"
                    }
                )
            )

    return documents
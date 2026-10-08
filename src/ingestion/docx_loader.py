from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document as LangChainDocument

from src.exception import ContentLoadError


def _iter_blocks(doc):
    # Read paragraphs and tables in the same order as the DOCX file.
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def load_docx(file_path: str):
    try:
        doc = Document(file_path)

    except Exception as e:
        raise ContentLoadError(
            "Could not read the DOCX file. It may be corrupted."
        ) from e

    blocks = []

    for block in _iter_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()

            if text:
                blocks.append(text)

        else:
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_text = " | ".join(cell for cell in cells if cell)

                if row_text:
                    blocks.append(row_text)

    if not blocks:
        raise ContentLoadError("No readable text was found in the DOCX file.")

    return [
        LangChainDocument(
            page_content="\n\n".join(blocks),
            metadata={
                "source": file_path,
                "type": "docx",
            },
        )
    ]


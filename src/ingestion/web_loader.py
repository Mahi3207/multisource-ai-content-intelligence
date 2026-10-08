import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from src.exception import ContentLoadError


USER_AGENT = "Mozilla/5.0 (AI Content Intelligence)"
os.environ.setdefault("USER_AGENT", USER_AGENT)

NOISE_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "svg",
]


def load_website(url: str):
    try:
        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": USER_AGENT},
        )
        soup = loader.scrape()

    except Exception as e:
        raise ContentLoadError(
            "Could not load the website. Please check the URL."
        ) from e

    # Remove elements that don't contain useful page content.
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n").strip()

    if len(text) < 200:
        raise ContentLoadError(
            "The website does not contain enough readable text."
        )

    title = soup.title.get_text(strip=True) if soup.title else ""

    return [
        Document(
            page_content=text,
            metadata={
                "source": url,
                "type": "website",
                "title": title,
            },
        )
    ]


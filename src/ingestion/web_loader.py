from langchain_community.document_loaders import WebBaseLoader


def load_website(url: str):
    """
    Load textual content from a website URL.

    Args:
        url: Website URL.

    Returns:
        List of LangChain Document objects.
    """

    loader = WebBaseLoader(url)

    documents = loader.load()

    return documents
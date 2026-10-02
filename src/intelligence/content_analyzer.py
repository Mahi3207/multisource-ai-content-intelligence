from langchain_core.prompts import PromptTemplate


def create_analysis_prompt():
    """
    Create the prompt used for content intelligence analysis.
    """

    template = """
You are an AI content intelligence assistant.

Analyze the following content and return a structured analysis.

Content:
{content}

Provide the following sections:

1. Summary
Write a concise summary of the content.

2. Key Points
List the most important points from the content.

3. Main Topics
Identify the main topics discussed.

4. Important Concepts
Identify important concepts, terms, or ideas.

5. Insights
Provide useful insights that can be directly derived from the content.

Rules:
- Use only information present in the provided content.
- Do not invent facts.
- Keep the analysis clear and concise.
- Use bullet points where appropriate.
"""

    return PromptTemplate(
        template=template,
        input_variables=["content"]
    )


def analyze_content(documents, llm):
    """
    Analyze a collection of LangChain documents using the LLM.

    Args:
        documents: List of LangChain Document objects.
        llm: Initialized language model.

    Returns:
        Generated content analysis.
    """

    content = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = create_analysis_prompt()

    formatted_prompt = prompt.format(
        content=content
    )

    response = llm.invoke(formatted_prompt)

    return response.content
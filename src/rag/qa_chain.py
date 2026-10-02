from langchain_core.prompts import PromptTemplate


def create_qa_prompt():
    """
    Create the prompt used for RAG question answering.
    """

    template = """
You are an AI content intelligence assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided content."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def answer_question(question, retriever, llm):
    """
    Retrieve relevant content and generate an answer using the LLM.
    """

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = create_qa_prompt()

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content,
        "sources": documents
    }
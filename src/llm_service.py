import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.config import LLM_MODEL


def get_llm():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Copy .env.example to .env and add your key."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=api_key,
        temperature=0,
    )
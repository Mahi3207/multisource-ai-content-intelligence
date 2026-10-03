
import os
import sys
import tempfile
import streamlit as st

# Make project root accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.content_loader import load_content
from src.processing.text_processor import split_documents
from src.embeddings.embedding_service import get_embedding_model
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import create_retriever
from src.rag.qa_chain import answer_question
from src.intelligence.content_analyzer import analyze_content

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

st.set_page_config(
    page_title="AI Content Intelligence",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def initialize_models():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found.")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0
    )

    embedding_model = get_embedding_model()

    return llm, embedding_model


st.title("🧠 AI Content Intelligence")
st.caption("Analyze content and ask questions using AI.")

llm, embedding_model = initialize_models()

st.divider()

st.subheader("Upload or provide content")

input_method = st.radio(
    "Choose input method",
    ["Upload a file", "Enter a URL"],
    horizontal=True
)

uploaded_file = None
url = ""

if input_method == "Upload a file":
    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"]
    )
else:
    url = st.text_input(
        "Website or YouTube URL",
        placeholder="https://..."
    )

if st.button("Process Content", type="primary"):
    if uploaded_file is None and not url.strip():
        st.warning("Please upload a file or enter a URL.")
    else:
        try:
            with st.spinner("Processing content..."):
                if uploaded_file is not None:
                    suffix = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        documents = load_content(temp_path)
                    finally:
                        os.remove(temp_path)
                else:
                    documents = load_content(url)

                if not documents:
                    st.error("No text could be extracted from this source.")
                    st.stop()

                chunks = split_documents(documents)

                if not chunks:
                    st.error("No usable text chunks were generated.")
                    st.stop()

                vector_store = create_vector_store(
                    chunks,
                    embedding_model
                )

                analysis = analyze_content(
                    chunks,
                    llm
                )

                st.session_state.documents = documents
                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store
                st.session_state.retriever = create_retriever(
                    vector_store,
                    k=3
                )
                st.session_state.analysis = analysis
                st.session_state.processed = True

            st.success("Content processed successfully!")

        except Exception as e:
            st.error(f"Processing failed: {e}")


if st.session_state.get("processed", False):
    st.divider()
    st.subheader("Content Analysis")
    st.markdown(st.session_state.analysis)

    st.divider()
    st.subheader("Ask Questions")

    question = st.text_input(
        "Ask a question about your content",
        key="user_question"
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    result = answer_question(
                        question,
                        st.session_state.retriever,
                        llm
                    )

                st.markdown("### Answer")
                st.write(result["answer"])

                with st.expander("View retrieved sources"):
                    for i, document in enumerate(result["sources"]):
                        st.markdown(f"**Source {i + 1}**")
                        st.write(document.page_content)
                        st.caption(str(document.metadata))

            except Exception as e:
                st.error(f"Question answering failed: {e}")
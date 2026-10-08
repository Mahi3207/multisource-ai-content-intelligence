import logging
import os
import sys
import tempfile

import streamlit as st

# Make the project root available for src imports.
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.config import TOO_LARGE_MESSAGE
from src.embeddings.embedding_service import get_embedding_model
from src.exception import ContentIntelligenceError, ContentTooLargeError
from src.llm_service import get_llm
from src.pipeline import process_source, release_content
from src.rag.qa_chain import answer_question, format_source_label


logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="AI Content Intelligence",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def initialize_models():
    return get_llm(), get_embedding_model()


def clear_current_content():
    release_content(st.session_state.pop("content", None))
    st.session_state.pop("last_result", None)
    st.session_state["user_question"] = ""


def process(source, display_name=None):
    # A new source replaces the previous one.
    clear_current_content()

    progress_bar = st.progress(0.0, text="Starting...")

    def on_progress(message, fraction):
        progress_bar.progress(
            min(max(fraction, 0.0), 1.0),
            text=message,
        )

    try:
        with st.spinner(
            "Processing content... "
            "(videos without subtitles are transcribed first)"
        ):
            st.session_state["content"] = process_source(
                source,
                embedding_model,
                llm,
                display_name=display_name,
                progress=on_progress,
            )

        progress_bar.empty()
        st.success("Content processed successfully!")

    except ContentTooLargeError:
        progress_bar.empty()
        st.warning(TOO_LARGE_MESSAGE)

    except ContentIntelligenceError as error:
        progress_bar.empty()
        st.error(str(error))

    except ValueError as error:
        progress_bar.empty()
        st.error(str(error))

    except Exception:
        logger.exception("Unexpected error while processing content")
        progress_bar.empty()
        st.error(
            "Something went wrong while processing this content. "
            "Please try again or use a different source."
        )


st.title("🧠 AI Content Intelligence")
st.caption("Analyze content and ask questions using AI.")

try:
    llm, embedding_model = initialize_models()

except ValueError as error:
    st.error(str(error))
    st.stop()

except Exception:
    logger.exception("Could not initialise models")
    st.error(
        "The AI models could not be started. "
        "Please check your setup and try again."
    )
    st.stop()


st.divider()

st.subheader("Upload or provide content")

input_method = st.radio(
    "Choose input method",
    ["Upload a file", "Enter a URL"],
    horizontal=True,
)

uploaded_file = None
url = ""

if input_method == "Upload a file":
    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
    )
else:
    url = st.text_input(
        "Website or YouTube URL",
        placeholder="https://...",
    )


if st.button("Process Content", type="primary"):
    if uploaded_file is None and not url.strip():
        st.warning("Please upload a file or enter a URL.")

    elif uploaded_file is not None:
        file_name = os.path.basename(uploaded_file.name)

        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        ) as temp_dir:
            temp_path = os.path.join(temp_dir, file_name)

            with open(temp_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())

            process(temp_path, display_name=file_name)

    else:
        process(url.strip())


content = st.session_state.get("content")

if content is not None:
    st.divider()

    st.caption(
        f"Current content: **{content.name}** "
        f"({content.source_type}, "
        f"{content.total_chars:,} characters, "
        f"{len(content.chunks)} chunks)"
    )

    st.subheader("Content Analysis")
    st.markdown(content.analysis)

    st.divider()
    st.subheader("Ask Questions")

    question = st.text_input(
        "Ask a question about your content",
        key="user_question",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Generating answer..."):
                    st.session_state["last_result"] = {
                        "doc_id": content.doc_id,
                        "question": question,
                        **answer_question(
                            question,
                            content.retriever,
                            llm,
                            analysis=content.analysis,
                        ),
                    }

            except Exception:
                logger.exception("Question answering failed")
                st.session_state.pop("last_result", None)
                st.error(
                    "The question could not be answered right now. "
                    "Please try again."
                )

    result = st.session_state.get("last_result")

    # Don't display an answer from an older document.
    if result and result["doc_id"] == content.doc_id:
        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Retrieved Sources")

        if result["mode"] == "overview":
            st.caption(
                "This is a document-level question, so it was answered "
                "from the analysis of the entire document rather than "
                "from a few chunks."
            )

        with st.expander("View retrieved sources", expanded=True):
            for i, document in enumerate(result["sources"]):
                st.markdown(
                    f"**Source {i + 1}** — "
                    f"{format_source_label(document.metadata)}"
                )
                st.write(document.page_content)
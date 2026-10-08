import logging
import time

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    ANALYSIS_BATCH_CHARS,
    ANALYSIS_MERGE_CHARS,
    MAX_ANALYSIS_CHARS,
)
from src.exception import AnalysisError, ContentTooLargeError


logger = logging.getLogger(__name__)

SECTIONS = [
    "Summary",
    "Key Points",
    "Main Topics",
    "Important Concepts",
    "Insights",
]

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

NOTES_FORMAT = """\
## Summary
## Key Points
## Main Topics
## Important Concepts
## Insights"""

RULES = """\
Rules:
- Use only information from the given material.
- Do not invent facts.
- Be concise.
- Use bullet points for lists.
- Write in English."""


def create_chunk_analysis_prompt():
    template = f"""
You are an AI content intelligence assistant.

Analyse this part of a longer document and write short notes using these
headings:

{NOTES_FORMAT}

{RULES}

Content:
{{content}}
"""
    return PromptTemplate(template=template, input_variables=["content"])


def create_merge_prompt():
    template = f"""
You are an AI content intelligence assistant.

Merge the following notes from different parts of the same document.

Use these headings:

{NOTES_FORMAT}

Combine repeated points and keep important information.

{RULES}

Notes:
{{notes}}
"""
    return PromptTemplate(template=template, input_variables=["notes"])


def create_final_analysis_prompt():
    template = f"""
You are an AI content intelligence assistant.

Create a final report from the material below.

Use exactly these headings:

## Summary
Give a concise overall summary.

## Key Points
List the most important points.

## Main Topics
List the main topics discussed.

## Important Concepts
Explain important concepts, terms, or ideas.

## Insights
Give useful insights that can be derived from the content.

{RULES}

{{material_kind}}:
{{material}}
"""
    return PromptTemplate(
        template=template,
        input_variables=["material_kind", "material"],
    )


def _join_text(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )


def check_size(documents):
    total = len(_join_text(documents))

    if total > MAX_ANALYSIS_CHARS:
        raise ContentTooLargeError(
            f"Content has {total:,} characters. "
            f"The limit is {MAX_ANALYSIS_CHARS:,}."
        )

    return total


def _split_batches(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ANALYSIS_BATCH_CHARS,
        chunk_overlap=0,
    )

    return splitter.split_text(text)


def _group_notes(notes, max_chars):
    groups = []
    current = []
    size = 0

    for note in notes:
        if len(current) >= 2 and size + len(note) > max_chars:
            groups.append(current)
            current = []
            size = 0

        current.append(note)
        size += len(note)

    if current:
        groups.append(current)

    return groups


def _has_all_sections(text):
    text = text.lower()
    return all(section.lower() in text for section in SECTIONS)


def _invoke(llm, prompt, require_sections=False):
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = llm.invoke(prompt).content

            if isinstance(response, list):
                response = " ".join(
                    part if isinstance(part, str) else part.get("text", "")
                    for part in response
                )

            text = (response or "").strip()

            if not text:
                raise ValueError("The model returned an empty response.")

            if require_sections and not _has_all_sections(text):
                raise ValueError("The response is missing required sections.")

            return text

        except Exception as e:
            last_error = e

            logger.warning(
                "LLM call failed (%d/%d): %s",
                attempt + 1,
                MAX_ATTEMPTS,
                e,
            )

            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))

    raise AnalysisError(
        "The AI service could not finish the analysis. Please try again."
    ) from last_error


def _report(progress, message, fraction):
    if progress:
        progress(message, fraction)


def analyze_content(documents, llm, progress=None):
    if not documents:
        raise AnalysisError("There is no content to analyse.")

    check_size(documents)

    text = _join_text(documents)
    batches = _split_batches(text)
    final_prompt = create_final_analysis_prompt()

    # Small documents can be analysed in one call.
    if len(batches) == 1:
        _report(progress, "Analysing content...", 0.1)

        result = _invoke(
            llm,
            final_prompt.format(
                material_kind="Document content",
                material=batches[0],
            ),
            require_sections=True,
        )

        _report(progress, "Analysis complete", 1.0)
        return result

    # Analyse each part separately first.
    chunk_prompt = create_chunk_analysis_prompt()
    notes = []

    for index, batch in enumerate(batches, start=1):
        _report(
            progress,
            f"Analysing part {index} of {len(batches)}...",
            0.8 * (index - 1) / len(batches),
        )

        notes.append(
            _invoke(
                llm,
                chunk_prompt.format(content=batch),
            )
        )

    # Merge the smaller results if they are still too large.
    merge_prompt = create_merge_prompt()
    round_number = 0

    while (
        len(notes) > 1
        and sum(len(note) for note in notes) > ANALYSIS_MERGE_CHARS
    ):
        round_number += 1

        _report(
            progress,
            f"Combining partial results (round {round_number})...",
            0.85,
        )

        merged = []

        for group in _group_notes(notes, ANALYSIS_MERGE_CHARS):
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(
                    _invoke(
                        llm,
                        merge_prompt.format(
                            notes="\n\n".join(group)
                        ),
                    )
                )

        notes = merged

    # Create the final report.
    _report(progress, "Writing final report...", 0.95)

    material = "\n\n---\n\n".join(notes)

    result = _invoke(
        llm,
        final_prompt.format(
            material_kind="Partial analyses",
            material=material,
        ),
        require_sections=True,
    )

    _report(progress, "Analysis complete", 1.0)

    return result


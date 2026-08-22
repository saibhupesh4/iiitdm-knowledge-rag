"""
citations.py

Builds the user-facing citation list for a final answer:
    [1] Hostel Rules and Regulations 2025-26 - Page 3
    [2] Hostel Regulations - Page 7

Rule: a citation must always correspond to an actual retrieved chunk.
Never invent page numbers or document titles. If the answer could not be
supported, the caller (pipeline.py) should use the standard refusal
message rather than fabricating citations:
    "I couldn't find sufficient information in the available IIITDM documents."

Input:  evidence chunks (list of RerankedCandidate) that were actually
        used to support the final answer (i.e. after verification.py)
Output: list of formatted citation strings, in source order
"""

from __future__ import annotations

from dataclasses import dataclass

from src.reranker import RerankedCandidate


@dataclass
class Citation:
    index: int
    title: str
    page: int
    source_filename: str


def build_citations(chunks: list[RerankedCandidate]) -> list[Citation]:
    """
    Convert evidence chunks into Citation objects, numbered in the order
    given (which should already reflect relevance / usage order).
    """
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for chunk in chunks:
        title = chunk.metadata.get("title", chunk.metadata.get("source", "Unknown document"))
        page = chunk.metadata.get("page", 0)
        source_filename = chunk.metadata.get("source", "unknown")

        key = (title, page)
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            Citation(
                index=len(citations) + 1,
                title=title,
                page=page,
                source_filename=source_filename,
            )
        )

    return citations


def format_citations(citations: list[Citation]) -> str:
    """
    Render Citation objects as the display string shown in Streamlit.
    """
    return "\n".join(f"[{c.index}] {c.title} - Page {c.page}" for c in citations)


if __name__ == "__main__":
    print("build_citations() requires evidence chunks; run via src.pipeline.answer_question().")

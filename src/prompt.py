"""
prompt.py

Builds the grounded prompt sent to the local LLM through LM Studio.

Architecture:

    Reranked evidence
            ↓
       prompt.py
            ↓
      system prompt
      + user prompt
            ↓
         llm.py
            ↓
        LM Studio

The LLM is responsible only for producing a concise answer from the
retrieved evidence.

The LLM must:

- use only the supplied evidence
- preserve conditions and exceptions
- avoid unsupported inference
- avoid document/version speculation
- cite factual statements
- return only the final answer

Input:
    question: str
    chunks: list[RerankedCandidate]

Output:
    tuple[str, str]
        (system_prompt, user_prompt)
"""

from __future__ import annotations

from src.reranker import RerankedCandidate


# =============================================================
# System Prompt
# =============================================================

SYSTEM_PROMPT = """
You are the final answer-generation component of an IIITDM Kancheepuram
document question-answering system.

Your task is to answer the user's question using ONLY the retrieved
evidence supplied in the user message.

STRICT GROUNDING RULES

1. Use ONLY the supplied evidence.
2. Do NOT use outside knowledge.
3. Do NOT invent facts, dates, numbers, names, procedures, or requirements.
4. Do NOT infer information that is not explicitly supported.
5. Preserve all conditions, exceptions, limits, and special cases exactly
   as supported by the evidence.
6. Do NOT generalize a rule from a special case to every situation.
7. If a source describes a rule only for a particular situation, keep that
   condition in the answer.
8. Do NOT speculate about which document is newer, older, official, updated,
   superseded, or contradictory unless the evidence explicitly states that.
9. Do NOT try to reconcile conflicting documents using your own judgment.
10. If two sources provide different information, report the difference
    cautiously rather than inventing an explanation.

ANSWER FORMAT

11. Return ONLY the final answer.
12. Do NOT provide reasoning or analysis.
13. Do NOT describe the retrieved evidence.
14. Do NOT say "I reviewed the sources".
15. Do NOT create sections such as:
       - Analysis
       - Evidence Analysis
       - Overview
       - Conclusion
       - Notes
       - Sources Reviewed
16. Do NOT add a document-comparison discussion unless explicitly asked.
17. Keep the answer concise and directly answer the question.
18. Prefer a short paragraph or a small number of bullet points.
19. Every factual statement must have one or more source citations.
20. Use citations exactly as:
       [Source 1]
       [Source 2]
       [Source 1][Source 2]

INSUFFICIENT EVIDENCE

21. If the retrieved evidence does not answer the question, say:

    "The available IIITDM documents do not provide enough information
    to answer this question reliably."

22. Do not try to answer an unsupported part using general knowledge.

IMPORTANT

Paraphrasing is allowed.

However, the meaning, scope, conditions, and exceptions of the evidence
must remain unchanged.

Return only the final answer.
""".strip()


# =============================================================
# Evidence Formatting
# =============================================================

def format_evidence_context(
    chunks: list[RerankedCandidate] | None,
) -> str:
    """
    Format reranked evidence into a numbered source context.

    Each usable chunk receives one stable [Source N] label.

    Empty or invalid chunks are skipped safely.
    """

    if not chunks:
        return "[No evidence was retrieved.]"

    blocks: list[str] = []

    source_number = 1

    for chunk in chunks:

        if chunk is None:
            continue

        text = getattr(
            chunk,
            "text",
            "",
        )

        if not isinstance(text, str):
            text = str(text or "")

        text = text.strip()

        # Ignore completely empty evidence chunks.
        if not text:
            continue

        metadata = getattr(
            chunk,
            "metadata",
            None,
        ) or {}

        source = (
            metadata.get("source")
            or metadata.get("title")
            or metadata.get("document")
            or "Unknown document"
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        section = (
            metadata.get("section")
            or metadata.get("heading")
            or "Not specified"
        )

        block = (
            f"[Source {source_number}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Evidence:\n"
            f"{text}"
        )

        blocks.append(block)

        source_number += 1

    if not blocks:
        return "[No usable evidence was retrieved.]"

    separator = (
        "\n\n"
        + "=" * 70
        + "\n\n"
    )

    return separator.join(blocks)


# =============================================================
# Prompt Construction
# =============================================================

def build_prompt(
    question: str,
    chunks: list[RerankedCandidate] | None,
) -> tuple[str, str]:
    """
    Build the system and user prompts for LM Studio.

    Parameters
    ----------
    question:
        User's original question.

    chunks:
        Final reranked evidence.

    Returns
    -------
    tuple[str, str]
        system_prompt, user_prompt
    """

    # ---------------------------------------------------------
    # Validate question.
    # ---------------------------------------------------------

    if not isinstance(question, str):
        raise ValueError(
            "question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "question cannot be empty."
        )

    # ---------------------------------------------------------
    # Format evidence.
    # ---------------------------------------------------------

    evidence_context = format_evidence_context(
        chunks
    )

    # ---------------------------------------------------------
    # User prompt
    # ---------------------------------------------------------

    user_prompt = f"""
RETRIEVED EVIDENCE
==================

{evidence_context}

USER QUESTION
=============

{question}

TASK
====

Answer the user question using ONLY the retrieved evidence.

Before producing the final answer, internally check:

- Is every factual statement supported?
- Am I preserving conditions and exceptions?
- Am I avoiding unsupported inference?
- Am I avoiding assumptions about document versions?
- Am I citing the relevant source for every factual statement?

Do not output this checking process.

OUTPUT REQUIREMENTS
===================

Return ONLY the final answer.

Keep it concise.

Do not provide:
- analysis
- reasoning
- an evidence review
- a source-by-source discussion
- document-version speculation
- a conclusion section
- a notes section

Cite factual statements using [Source N].
""".strip()

    return SYSTEM_PROMPT, user_prompt


# =============================================================
# Standalone Test
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IIITDM Knowledge RAG - Prompt Module")
    print("=" * 60)

    print()
    print(
        "Prompt builder loaded successfully."
    )

    print()
    print(
        "Use the complete pipeline with:"
    )

    print(
        "    python -m src.pipeline"
    )

    print()
    print("=" * 60)
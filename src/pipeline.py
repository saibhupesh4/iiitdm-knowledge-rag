"""
pipeline.py

Top-level orchestration for the IIITDM Knowledge RAG system.

Pipeline:

    User Question
        ↓
    Query Routing
        ├── General question → local LM Studio → answer (no citations)
        └── IIITDM/document question
                ↓
            Query Understanding
        ↓
    Hybrid Retrieval
        ├── Dense Retrieval
        └── BM25
        ↓
    RRF Fusion
        ↓
    Top-K Hybrid Candidates
        ↓
    Cross-Encoder Reranker
        ↓
    Evidence Sufficiency Gate
        │
        ├── INSUFFICIENT
        │       ↓
        │   Query Reformulation
        │       ↓
        │   Retrieve + Rerank again
        │
        └── SUFFICIENT
                ↓
            Prompt Builder
                ↓
            LM Studio
                ↓
            Verification
                │
                ├── Unsupported
                │       ↓
                │   Regenerate once
                │
                └── Supported
                        ↓
                    Citations
                        ↓
                   Final Answer

This module is the ONLY module app.py should call for answering questions.

Important:

- No external LLM API fallback.
- BM25 index is loaded lazily and cached.
- General questions bypass retrieval; document questions retrieve before generation.
- Evidence is checked before generation.
- Generated answers are verified before being returned.
- Refusal reasons are kept separate from the user-facing refusal text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field



# ============================================================
# Project imports
# ============================================================

from src.config import (
    MAX_QUERY_REFORMULATION_RETRIES,
    BM25_INDEX_PATH,
)

from src.query_understanding import (
    detect_category,
    is_app_help_input,
    is_document_question,
)

from src.hybrid_retrieval import (
    HybridCandidate,
    hybrid_search,
)

from src.reranker import (
    RerankedCandidate,
    rerank,
)

from src.evidence_gate import (
    EvidenceVerdict,
    EvidenceSufficiency,
    check_sufficiency,
)

from src.query_reformulation import (
    reformulate_query,
)

from src.prompt import (
    build_prompt,
)

from src.llm import (
    generate,
    LMStudioUnavailableError,
    LMStudioConfigurationError,
    LMStudioGenerationError,
)

from src.verification import (
    VerificationResult,
    verify_answer,
)

from src.citations import (
    Citation,
    build_citations,
)

from src.bm25 import (
    BM25Index,
)


# ============================================================
# Pipeline result
# ============================================================

@dataclass
class PipelineResult:
    """
    Final object returned by answer_question().

    Streamlit should use this object instead of directly calling
    individual retrieval or generation modules.
    """

    question: str
    category: str

    answer: str

    citations: list[Citation]

    # Retrieval diagnostics
    candidates_retrieved: int
    candidates_after_rerank: int

    # Query reformulation diagnostics
    query_was_reformulated: bool
    reformulated_query: str | None

    # Evidence diagnostics
    evidence_verdict: EvidenceVerdict

    # Verification diagnostics
    verification_result: VerificationResult | None

    # Keep the exact reranked chunks available for source auditing.
    evidence: list[RerankedCandidate] = field(default_factory=list)

    # Final status
    refused: bool = False
    refusal_reason: str | None = None
    answer_mode: str = "rag"


# ============================================================
# User-facing refusal messages
# ============================================================

EVIDENCE_REFUSAL_MESSAGE = (
    "I couldn't find sufficient information in the available "
    "IIITDM documents to answer this question reliably."
)

VERIFICATION_REFUSAL_MESSAGE = (
    "I found relevant IIITDM documents, but I could not verify "
    "the generated answer against that evidence reliably."
)

EMPTY_ANSWER_MESSAGE = (
    "The local language model did not return a usable answer."
)

APP_HELP_MESSAGE = (
    "I'm IIITDM Knowledge RAG, a document-grounded assistant for official "
    "IIITDM Kancheepuram documents. I can help with academics, hostel, "
    "placements, and institute information."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. Answer the user's "
    "question clearly and concisely. Be honest when you are unsure. "
    "Do not claim that your answer comes from IIITDM documents."
)


def _general_result(question: str, answer: str) -> PipelineResult:
    """Return a local-model answer without document citations or grounding claims."""
    return PipelineResult(
        question=question,
        category="general",
        answer=answer,
        citations=[],
        candidates_retrieved=0,
        candidates_after_rerank=0,
        query_was_reformulated=False,
        reformulated_query=None,
        evidence_verdict=check_sufficiency([]),
        verification_result=None,
        answer_mode="general",
    )


# ============================================================
# Cached BM25 index
# ============================================================

_bm25_index: BM25Index | None = None


# ============================================================
# BM25 loader
# ============================================================

def _get_bm25_index() -> BM25Index:
    """
    Load the persisted BM25 index once and reuse it.

    Raises
    ------
    RuntimeError
        If the persisted index does not exist.
    """

    global _bm25_index

    if _bm25_index is not None:
        return _bm25_index

    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(
            f"BM25 index was not found:\n"
            f"{BM25_INDEX_PATH}\n\n"
            "Build the indexes first using:\n"
            "python scripts/rebuild_index.py"
        )

    try:
        _bm25_index = BM25Index.load(
            str(BM25_INDEX_PATH)
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to load the persisted BM25 index.\n"
            f"Path: {BM25_INDEX_PATH}\n"
            f"Reason: {exc}"
        ) from exc

    return _bm25_index


# ============================================================
# Retrieval + reranking
# ============================================================

def _retrieve_and_rerank(
    query: str,
    bm25_index: BM25Index,
) -> tuple[
    list[HybridCandidate],
    list[RerankedCandidate],
]:
    """
    Run:

        Dense + BM25 + RRF
                ↓
          Hybrid candidates
                ↓
          Cross-encoder
                ↓
          Reranked evidence
    """

    if not query or not query.strip():
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    hybrid_candidates = hybrid_search(
        query.strip(),
        bm25_index,
    )

    # Never pass None into the reranker.
    if hybrid_candidates is None:
        hybrid_candidates = []

    reranked_candidates = rerank(
        query.strip(),
        hybrid_candidates,
    )

    if reranked_candidates is None:
        reranked_candidates = []

    return (
        hybrid_candidates,
        reranked_candidates,
    )


# ============================================================
# Diagnostic helpers
# ============================================================

def _print_evidence_diagnostics(
    verdict: EvidenceVerdict,
    candidates: list[RerankedCandidate],
) -> None:
    """
    Print evidence diagnostics during CLI testing.

    This makes it obvious whether refusal happened before or after
    LLM generation.
    """

    print()
    print("=" * 60)
    print("EVIDENCE GATE")
    print("=" * 60)

    print(
        "Status:",
        verdict.status.value,
    )

    print(
        "Qualifying chunks:",
        verdict.num_qualifying_chunks,
    )

    print(
        "Threshold:",
        verdict.threshold_used,
    )

    print(
        "Minimum required:",
        verdict.min_chunks_required,
    )

    print()
    print("Reranked evidence:")

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        metadata = candidate.metadata or {}

        print()
        print(f"[{index}]")
        print(
            "Relevance:",
            f"{candidate.relevance_score:.6f}",
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "?",
            ),
        )

        text = (
            candidate.text or ""
        ).strip()

        if len(text) > 500:
            text = text[:500] + "..."

        print(
            "Text:",
            text,
        )


def _print_verification_diagnostics(
    draft_answer: str,
    result: VerificationResult,
) -> None:
    """
    Print generation and verification diagnostics.
    """

    print()
    print("=" * 60)
    print("LLM DRAFT ANSWER")
    print("=" * 60)
    print(
        draft_answer
        if draft_answer
        else "[EMPTY]"
    )

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(
        "Total claims:",
        result.total_count,
    )

    print(
        "Supported claims:",
        result.supported_count,
    )

    print(
        "Unsupported claims:",
        result.unsupported_count,
    )

    print(
        "Support ratio:",
        f"{result.support_ratio:.4f}",
    )

    for index, claim in enumerate(
        result.claims,
        start=1,
    ):

        print()
        print(f"Claim {index}:")
        print(
            "Text:",
            claim.claim_text,
        )

        print(
            "Status:",
            claim.status.value,
        )

        print(
            "Score:",
            getattr(
                claim,
                "support_score",
                getattr(
                    claim,
                    "score",
                    0.0,
                ),
            ),
        )

        print(
            "Supporting sources:",
            claim.supporting_source_indices,
        )


# ============================================================
# Main pipeline
# ============================================================

def answer_question(
    question: str,
    *,
    debug: bool = False,
) -> PipelineResult:
    """
    Run the complete RAG pipeline.

    Parameters
    ----------
    question:
        User question.

    debug:
        Print evidence, LLM, and verification diagnostics.

    Returns
    -------
    PipelineResult
        Final answer and retrieval/grounding diagnostics.

    Raises
    ------
    ValueError
        Invalid question.

    RuntimeError
        Missing or broken BM25 index.

    LMStudioUnavailableError
        LM Studio cannot be reached.

    LMStudioConfigurationError
        LM Studio configuration is invalid.

    LMStudioGenerationError
        LM Studio generation failed.
    """

    # ========================================================
    # 1. Validate question
    # ========================================================

    if not isinstance(question, str):
        raise ValueError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    # Greetings and questions about this app need no model or document search.
    if is_app_help_input(question):
        return _general_result(question, APP_HELP_MESSAGE)

    # Ordinary questions go to local chat. IIITDM and document topics
    # continue into retrieval, evidence checks, and source citations.
    if not is_document_question(question):
        answer = generate(
            GENERAL_SYSTEM_PROMPT,
            question,
            temperature=0.2,
            max_tokens=1000,
        )
        return _general_result(question, answer)

    # ========================================================
    # 2. Load BM25
    # ========================================================

    bm25_index = _get_bm25_index()

    # ========================================================
    # 3. Query understanding
    # ========================================================

    category = detect_category(
        question
    )

    # Protect the pipeline from a broken classifier returning None.
    if not category:
        category = "unknown"

    # ========================================================
    # 4. Initial retrieval
    # ========================================================

    (
        hybrid_candidates,
        reranked_candidates,
    ) = _retrieve_and_rerank(
        question,
        bm25_index,
    )

    # ========================================================
    # 5. Evidence gate
    # ========================================================

    verdict = check_sufficiency(
        reranked_candidates
    )

    if debug:
        _print_evidence_diagnostics(
            verdict,
            reranked_candidates,
        )

    # ========================================================
    # 6. Query reformulation
    # ========================================================

    query_was_reformulated = False

    reformulated_query: str | None = None

    retries_allowed = max(
        0,
        int(
            MAX_QUERY_REFORMULATION_RETRIES
        ),
    )

    retries_used = 0

    while (
        verdict.status
        is EvidenceSufficiency.INSUFFICIENT
        and retries_used < retries_allowed
    ):

        retries_used += 1

        try:
            reformulated_query = (
                reformulate_query(
                    question,
                    category,
                )
            )

        except Exception:
            reformulated_query = None

        if not reformulated_query:
            break

        reformulated_query = (
            reformulated_query.strip()
        )

        if not reformulated_query:
            break

        # Prevent the reformulation module from returning the exact
        # original query forever.
        if (
            reformulated_query.lower()
            == question.lower()
        ):
            break

        query_was_reformulated = True

        (
            hybrid_candidates,
            reranked_candidates,
        ) = _retrieve_and_rerank(
            reformulated_query,
            bm25_index,
        )

        verdict = check_sufficiency(
            reranked_candidates
        )

        if debug:
            print()
            print(
                f"[reformulation] Retry {retries_used}"
            )

            print(
                "[reformulation] Query:",
                reformulated_query,
            )

            _print_evidence_diagnostics(
                verdict,
                reranked_candidates,
            )

    # ========================================================
    # 7. Refuse only for insufficient retrieval evidence
    # ========================================================

    if (
        verdict.status
        is EvidenceSufficiency.INSUFFICIENT
    ):

        return PipelineResult(
            question=question,
            category=category,

            answer=EVIDENCE_REFUSAL_MESSAGE,

            citations=[],

            candidates_retrieved=len(
                hybrid_candidates
            ),

            candidates_after_rerank=len(
                reranked_candidates
            ),

            query_was_reformulated=
                query_was_reformulated,

            reformulated_query=
                reformulated_query,

            evidence_verdict=
                verdict,

            evidence=reranked_candidates,

            verification_result=None,

            refused=True,

            refusal_reason=(
                "Evidence sufficiency gate returned "
                "INSUFFICIENT."
            ),
        )

    # ========================================================
    # 8. Build grounded prompt
    # ========================================================

    (
        system_prompt,
        user_prompt,
    ) = build_prompt(
        question,
        reranked_candidates,
    )

    # ========================================================
    # 9. First LLM generation
    # ========================================================

    draft_answer = generate(
        system_prompt,
        user_prompt,
        temperature=0.0,
        max_tokens=2000,
    )

    # ========================================================
    # 10. Validate generated answer
    # ========================================================

    if not isinstance(
        draft_answer,
        str,
    ):

        raise LMStudioGenerationError(
            "LM Studio returned a non-string answer."
        )

    draft_answer = draft_answer.strip()

    if not draft_answer:

        return PipelineResult(
            question=question,
            category=category,

            answer=EMPTY_ANSWER_MESSAGE,

            citations=[],

            candidates_retrieved=len(
                hybrid_candidates
            ),

            candidates_after_rerank=len(
                reranked_candidates
            ),

            query_was_reformulated=
                query_was_reformulated,

            reformulated_query=
                reformulated_query,

            evidence_verdict=
                verdict,

            evidence=reranked_candidates,

            verification_result=None,

            refused=True,

            refusal_reason=(
                "LM Studio returned an empty answer."
            ),
        )

    # ========================================================
    # 11. Verify first answer
    # ========================================================

    verification_result = verify_answer(
        draft_answer,
        reranked_candidates,
    )

    if debug:
        _print_verification_diagnostics(
            draft_answer,
            verification_result,
        )

    # ========================================================
    # 12. One regeneration attempt
    # ========================================================

    #
    # Important:
    #
    # Do NOT require the first answer to have perfect lexical support.
    # The verifier may reject valid paraphrases.
    #
    # If the first verification is weak, ask the same grounded prompt
    # again with slightly different sampling.
    #

    first_ratio = (
        verification_result.support_ratio
    )

    needs_retry = (
        verification_result.total_count == 0
        or not verification_result.all_supported
    )

    if needs_retry:

        retry_answer: str | None = None

        try:

            retry_answer = generate(
                system_prompt,
                user_prompt,
                temperature=0.1,
                max_tokens=2000,
            )

        except LMStudioUnavailableError:

            # Do not replace a valid first answer merely because
            # LM Studio disappeared during the optional retry.
            retry_answer = None

        if (
            isinstance(
                retry_answer,
                str,
            )
            and retry_answer.strip()
        ):

            retry_answer = (
                retry_answer.strip()
            )

            retry_verification = verify_answer(
                retry_answer,
                reranked_candidates,
            )

            retry_ratio = (
                retry_verification.support_ratio
            )

            # Prefer:
            #
            # 1. fully supported retry
            # 2. higher support ratio
            # 3. more supported claims
            #

            retry_is_better = False

            if retry_verification.all_supported:
                retry_is_better = True

            elif retry_ratio > first_ratio:
                retry_is_better = True

            elif (
                retry_ratio == first_ratio
                and
                retry_verification.supported_count
                >
                verification_result.supported_count
            ):
                retry_is_better = True

            if retry_is_better:

                draft_answer = retry_answer

                verification_result = (
                    retry_verification
                )

                if debug:

                    print()
                    print(
                        "[verification] "
                        "Using regenerated answer."
                    )

                    _print_verification_diagnostics(
                        draft_answer,
                        verification_result,
                    )

    # ========================================================
    # 13. Final verification decision
    # ========================================================

    #
    # IMPORTANT:
    #
    # Prefer the verifier's own acceptance property when available.
    # This makes pipeline.py compatible with the improved verifier.
    #

    is_acceptable = False

    if hasattr(
        verification_result,
        "is_acceptable",
    ):
        is_acceptable = bool(
            verification_result.is_acceptable
        )

    elif hasattr(
        verification_result,
        "sufficiently_supported",
    ):
        is_acceptable = bool(
            verification_result.sufficiently_supported
        )

    else:
        # Conservative fallback for older verification.py versions.
        is_acceptable = bool(
            verification_result.all_supported
            and verification_result.total_count > 0
        )

    # ========================================================
    # 14. Refuse unsupported generated answer
    # ========================================================

    if not is_acceptable:

        return PipelineResult(
            question=question,
            category=category,

            answer=VERIFICATION_REFUSAL_MESSAGE,

            citations=[],

            candidates_retrieved=len(
                hybrid_candidates
            ),

            candidates_after_rerank=len(
                reranked_candidates
            ),

            query_was_reformulated=
                query_was_reformulated,

            reformulated_query=
                reformulated_query,

            evidence_verdict=
                verdict,

            evidence=reranked_candidates,

            verification_result=
                verification_result,

            refused=True,

            refusal_reason=(
                "Retrieved evidence was sufficient, but the "
                "generated answer could not be verified reliably "
                "against that evidence."
            ),
        )

    # ========================================================
    # 15. Build citations
    # ========================================================

    try:

        citations = build_citations(
            reranked_candidates
        )

    except Exception:
        # Citation construction should never destroy an otherwise
        # successfully grounded answer.
        citations = []

    # ========================================================
    # 16. Return successful result
    # ========================================================

    return PipelineResult(
        question=question,
        category=category,

        answer=draft_answer,

        citations=citations,

        candidates_retrieved=len(
            hybrid_candidates
        ),

        candidates_after_rerank=len(
            reranked_candidates
        ),

        query_was_reformulated=
            query_was_reformulated,

        reformulated_query=
            reformulated_query,

        evidence_verdict=
            verdict,

        evidence=reranked_candidates,

        verification_result=
            verification_result,

        refused=False,

        refusal_reason=None,
    )


# ============================================================
# CLI test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IIITDM Knowledge RAG - Pipeline Test")
    print("=" * 60)

    test_question = (
        "What are the hostel leave rules?"
    )

    print()
    print("[test] Question:")
    print(test_question)

    print()

    try:

        result = answer_question(
            test_question,
            debug=True,
        )

        print()
        print("=" * 60)
        print("PIPELINE RESULT")
        print("=" * 60)

        print()
        print(
            "Category:",
            result.category,
        )

        print(
            "Candidates retrieved:",
            result.candidates_retrieved,
        )

        print(
            "After reranking:",
            result.candidates_after_rerank,
        )

        print(
            "Query reformulated:",
            result.query_was_reformulated,
        )

        if result.reformulated_query:
            print(
                "Reformulated query:",
                result.reformulated_query,
            )

        print()
        print(
            "Evidence status:",
            result.evidence_verdict.status.value,
        )

        print(
            "Qualifying evidence:",
            result.evidence_verdict.num_qualifying_chunks,
        )

        if result.verification_result:

            print()
            print(
                "Verification support ratio:",
                f"{result.verification_result.support_ratio:.4f}",
            )

            print(
                "Supported claims:",
                result.verification_result.supported_count,
            )

            print(
                "Total claims:",
                result.verification_result.total_count,
            )

        print()
        print(
            "Refused:",
            result.refused,
        )

        if result.refusal_reason:

            print(
                "Refusal reason:",
                result.refusal_reason,
            )

        print()
        print("Answer:")
        print("-" * 60)
        print(result.answer)

        print()
        print("Citations:")
        print("-" * 60)

        if result.citations:

            for citation in result.citations:
                print(citation)

        else:

            print("No citations.")

        print()
        print("=" * 60)
        print("PIPELINE TEST COMPLETED")
        print("=" * 60)

    except LMStudioUnavailableError as exc:

        print()
        print("[LM STUDIO UNAVAILABLE]")
        print(exc)

    except LMStudioConfigurationError as exc:

        print()
        print("[LM STUDIO CONFIGURATION ERROR]")
        print(exc)

    except LMStudioGenerationError as exc:

        print()
        print("[LM STUDIO GENERATION ERROR]")
        print(exc)

    except Exception as exc:

        print()
        print("[ERROR] Pipeline failed.")
        print(
            f"{type(exc).__name__}: {exc}"
        )

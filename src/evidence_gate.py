"""
evidence_gate.py

Evidence Sufficiency Gate for the IIITDM Knowledge RAG pipeline.

The gate decides whether the reranked evidence is strong enough to
continue to answer generation.

Rule:

    SUFFICIENT

        if at least `min_chunks` unique valid candidates have

            relevance_score >= score_threshold

    otherwise

        INSUFFICIENT

The module is deliberately deterministic and explainable. It does not
generate scores and does not use an LLM.

Input:
    list[RerankedCandidate] | None

Output:
    EvidenceVerdict
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Iterable



from src.config import (
    EVIDENCE_SCORE_THRESHOLD,
    EVIDENCE_MIN_CHUNKS,
)


# ============================================================
# Evidence status
# ============================================================

class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


# ============================================================
# Result object
# ============================================================

@dataclass
class EvidenceVerdict:
    """
    Final result returned by the evidence sufficiency gate.
    """

    status: EvidenceSufficiency
    is_sufficient: bool

    num_candidates_checked: int
    num_valid_candidates: int
    num_invalid_candidates: int

    num_qualifying_chunks: int

    threshold_used: float
    min_chunks_required: int

    reason: str

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the verdict into a JSON/UI-friendly dictionary.
        """
        data = asdict(self)
        data["status"] = self.status.value
        return data


# ============================================================
# Validation helpers
# ============================================================

def _validate_threshold(score_threshold: float) -> float:
    """
    Validate and normalize the evidence score threshold.

    Expected range:
        0.0 <= threshold <= 1.0
    """

    try:
        threshold = float(score_threshold)
    except (TypeError, ValueError):
        raise ValueError(
            "score_threshold must be a numeric value."
        )

    if not math.isfinite(threshold):
        raise ValueError(
            "score_threshold must be finite."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "score_threshold must be between 0.0 and 1.0."
        )

    return threshold


def _validate_min_chunks(min_chunks: int) -> int:
    """
    Validate the minimum number of qualifying evidence chunks.
    """

    # Prevent bool because bool is a subclass of int.
    if isinstance(min_chunks, bool):
        raise ValueError(
            "min_chunks must be a positive integer."
        )

    try:
        value = int(min_chunks)
    except (TypeError, ValueError):
        raise ValueError(
            "min_chunks must be a positive integer."
        )

    if value <= 0:
        raise ValueError(
            "min_chunks must be greater than 0."
        )

    return value


def _safe_relevance_score(candidate: Any) -> float | None:
    """
    Safely extract a relevance_score.

    Returns:
        float  -> valid finite score
        None   -> missing or invalid score
    """

    try:
        score = getattr(candidate, "relevance_score", None)

        if score is None:
            return None

        score = float(score)

        if not math.isfinite(score):
            return None

        return score

    except (TypeError, ValueError):
        return None


def _candidate_identity(candidate: Any, fallback_index: int) -> str:
    """
    Return a stable identity for duplicate detection.

    Preferred:
        candidate.chunk_id

    Fallback:
        unique positional identifier
    """

    chunk_id = getattr(candidate, "chunk_id", None)

    if chunk_id is None:
        return f"__candidate_{fallback_index}"

    try:
        chunk_id = str(chunk_id).strip()
    except Exception:
        return f"__candidate_{fallback_index}"

    if not chunk_id:
        return f"__candidate_{fallback_index}"

    return chunk_id


def _normalize_candidates(
    candidates: Any,
) -> list[Any]:
    """
    Normalize candidate input.

    Accepted:
        None
        list
        tuple
        other iterable

    Rejected:
        strings
        bytes
        dictionaries

    Strings and dictionaries are rejected because iterating over them
    would incorrectly treat characters/keys as candidates.
    """

    if candidates is None:
        return []

    if isinstance(candidates, (str, bytes, dict)):
        raise TypeError(
            "candidates must be an iterable of candidate objects, "
            "not str, bytes, or dict."
        )

    try:
        return list(candidates)
    except TypeError:
        raise TypeError(
            "candidates must be an iterable of candidate objects "
            "or None."
        )


# ============================================================
# Main evidence sufficiency gate
# ============================================================

def check_sufficiency(
    candidates: Iterable[Any] | None,
    score_threshold: float = EVIDENCE_SCORE_THRESHOLD,
    min_chunks: int = EVIDENCE_MIN_CHUNKS,
) -> EvidenceVerdict:
    """
    Determine whether reranked evidence is sufficient.

    A candidate qualifies when:

        relevance_score >= score_threshold

    Duplicate chunk IDs are counted only once.

    Invalid candidates do not crash the RAG pipeline. Instead they are
    counted as invalid and ignored.

    Parameters
    ----------
    candidates:
        Iterable of objects containing at least:

            chunk_id
            relevance_score

        None is treated as no evidence.

    score_threshold:
        Required relevance threshold in [0, 1].

    min_chunks:
        Minimum number of unique qualifying chunks required.

    Returns
    -------
    EvidenceVerdict
    """

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    threshold = _validate_threshold(score_threshold)
    required_chunks = _validate_min_chunks(min_chunks)

    # --------------------------------------------------------
    # Normalize input
    # --------------------------------------------------------

    normalized_candidates = _normalize_candidates(candidates)

    num_candidates_checked = len(normalized_candidates)

    # --------------------------------------------------------
    # Empty evidence
    # --------------------------------------------------------

    if not normalized_candidates:
        return EvidenceVerdict(
            status=EvidenceSufficiency.INSUFFICIENT,
            is_sufficient=False,

            num_candidates_checked=0,
            num_valid_candidates=0,
            num_invalid_candidates=0,

            num_qualifying_chunks=0,

            threshold_used=threshold,
            min_chunks_required=required_chunks,

            reason="No reranked evidence candidates were available.",
        )

    # --------------------------------------------------------
    # Process candidates
    # --------------------------------------------------------

    seen_chunk_ids: set[str] = set()

    num_valid_candidates = 0
    num_invalid_candidates = 0
    num_qualifying_chunks = 0

    for index, candidate in enumerate(normalized_candidates):

        # --------------------------------------------
        # Validate score
        # --------------------------------------------

        score = _safe_relevance_score(candidate)

        if score is None:
            num_invalid_candidates += 1
            continue

        num_valid_candidates += 1

        # --------------------------------------------
        # Prevent duplicate evidence counting
        # --------------------------------------------

        identity = _candidate_identity(
            candidate,
            fallback_index=index,
        )

        if identity in seen_chunk_ids:
            continue

        seen_chunk_ids.add(identity)

        # --------------------------------------------
        # Check evidence threshold
        # --------------------------------------------

        if score >= threshold:
            num_qualifying_chunks += 1

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    is_sufficient = (
        num_qualifying_chunks >= required_chunks
    )

    status = (
        EvidenceSufficiency.SUFFICIENT
        if is_sufficient
        else EvidenceSufficiency.INSUFFICIENT
    )

    if is_sufficient:

        reason = (
            f"Evidence is sufficient: "
            f"{num_qualifying_chunks} unique chunk(s) meet or exceed "
            f"the relevance threshold of {threshold:.3f}; "
            f"{required_chunks} required."
        )

    elif num_valid_candidates == 0:

        reason = (
            "No valid reranker relevance scores were available."
        )

    else:

        reason = (
            f"Evidence is insufficient: "
            f"{num_qualifying_chunks} unique chunk(s) meet or exceed "
            f"the relevance threshold of {threshold:.3f}; "
            f"{required_chunks} required."
        )

    return EvidenceVerdict(
        status=status,
        is_sufficient=is_sufficient,

        num_candidates_checked=num_candidates_checked,
        num_valid_candidates=num_valid_candidates,
        num_invalid_candidates=num_invalid_candidates,

        num_qualifying_chunks=num_qualifying_chunks,

        threshold_used=threshold,
        min_chunks_required=required_chunks,

        reason=reason,
    )


# ============================================================
# Standalone tests
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IIITDM Knowledge RAG - Evidence Sufficiency Gate")
    print("=" * 60)

    # --------------------------------------------------------
    # Small dummy class for standalone testing
    # --------------------------------------------------------

    @dataclass
    class DummyCandidate:
        chunk_id: str
        relevance_score: float


    def run_test(
        name: str,
        candidates: Any,
        threshold: float = EVIDENCE_SCORE_THRESHOLD,
        minimum: int = EVIDENCE_MIN_CHUNKS,
    ) -> None:

        print()
        print(f"Test: {name}")

        try:

            result = check_sufficiency(
                candidates=candidates,
                score_threshold=threshold,
                min_chunks=minimum,
            )

            print(result.to_dict())

        except Exception as error:

            print(
                f"Expected/handled error: "
                f"{type(error).__name__}: {error}"
            )


    # ========================================================
    # TEST 1: None
    # ========================================================

    run_test(
        "None candidates",
        None,
    )


    # ========================================================
    # TEST 2: Empty list
    # ========================================================

    run_test(
        "Empty candidates",
        [],
    )


    # ========================================================
    # TEST 3: Insufficient evidence
    # ========================================================

    run_test(
        "Insufficient evidence",
        [
            DummyCandidate("chunk_1", 0.90),
            DummyCandidate("chunk_2", 0.30),
            DummyCandidate("chunk_3", 0.20),
        ],
    )


    # ========================================================
    # TEST 4: Exactly sufficient
    # ========================================================

    run_test(
        "Exactly sufficient",
        [
            DummyCandidate("chunk_1", 0.90),
            DummyCandidate("chunk_2", 0.80),
            DummyCandidate("chunk_3", 0.20),
        ],
    )


    # ========================================================
    # TEST 5: Score exactly at threshold
    # ========================================================

    run_test(
        "Score exactly at threshold",
        [
            DummyCandidate("chunk_1", 0.50),
            DummyCandidate("chunk_2", 0.50),
        ],
        threshold=0.50,
        minimum=2,
    )


    # ========================================================
    # TEST 6: Duplicate chunk IDs
    # ========================================================

    run_test(
        "Duplicate chunk IDs",
        [
            DummyCandidate("chunk_1", 0.90),
            DummyCandidate("chunk_1", 0.95),
            DummyCandidate("chunk_2", 0.80),
        ],
        threshold=0.50,
        minimum=2,
    )


    # ========================================================
    # TEST 7: Invalid score
    # ========================================================

    run_test(
        "Invalid candidate score",
        [
            DummyCandidate("chunk_1", 0.90),
            DummyCandidate("chunk_2", float("nan")),
            DummyCandidate("chunk_3", 0.80),
        ],
    )


    # ========================================================
    # TEST 8: Infinity score
    # ========================================================

    run_test(
        "Infinite score",
        [
            DummyCandidate("chunk_1", float("inf")),
            DummyCandidate("chunk_2", 0.80),
        ],
    )


    # ========================================================
    # TEST 9: Missing relevance_score
    # ========================================================

    class BrokenCandidate:
        def __init__(self):
            self.chunk_id = "broken_chunk"


    run_test(
        "Missing relevance score",
        [
            BrokenCandidate(),
            DummyCandidate("chunk_2", 0.90),
        ],
    )


    # ========================================================
    # TEST 10: Invalid threshold
    # ========================================================

    run_test(
        "Invalid threshold",
        [
            DummyCandidate("chunk_1", 0.90),
        ],
        threshold=1.5,
    )


    # ========================================================
    # TEST 11: Invalid minimum chunks
    # ========================================================

    run_test(
        "Invalid minimum chunks",
        [
            DummyCandidate("chunk_1", 0.90),
        ],
        minimum=0,
    )


    # ========================================================
    # TEST 12: Invalid input type
    # ========================================================

    run_test(
        "String instead of candidates",
        "this should not be treated as candidates",
    )


    print()
    print("=" * 60)
    print("EVIDENCE GATE EDGE-CASE TESTS COMPLETED")
    print("=" * 60)
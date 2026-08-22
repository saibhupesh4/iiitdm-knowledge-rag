"""
hybrid_retrieval.py

Hybrid retrieval using Reciprocal Rank Fusion (RRF).

Combines:
    1. Dense semantic retrieval
    2. BM25 lexical retrieval

RRF combines document ranks rather than raw retrieval scores.

Formula:

    RRF(d) = sum(
        1 / (k + rank_i(d))
    )

A document appearing in both retrieval results receives contributions
from both retrievers.

Advantages:
    - No score normalization required
    - Robust to different score scales
    - Works when a document appears in only one retriever
    - Deterministic ranking
    - Independent of dense/BM25 score distributions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ============================================================
# Optional LangSmith tracing
# ============================================================

try:
    from langsmith import traceable

except ImportError:  # pragma: no cover

    def traceable(*args, **kwargs):
        """
        Fallback decorator when LangSmith is not installed.

        Supports both:

            @traceable

        and:

            @traceable(...)
        """

        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


# ============================================================
# Project imports
# ============================================================

from src.config import (
    DENSE_TOP_K,
    BM25_TOP_K,
    HYBRID_TOP_K,
    RRF_K,
)

from src.dense_retrieval import (
    dense_search,
    DenseCandidate,
)

from src.bm25 import (
    BM25Index,
    BM25Candidate,
)


# ============================================================
# Data structure
# ============================================================

@dataclass
class HybridCandidate:
    """
    Candidate produced after Reciprocal Rank Fusion.
    """

    chunk_id: str

    # Final Reciprocal Rank Fusion score.
    rrf_score: float

    # Rank in dense retrieval.
    # None means the chunk did not appear in dense retrieval.
    dense_rank: int | None

    # Rank in BM25 retrieval.
    # None means the chunk did not appear in BM25 retrieval.
    bm25_rank: int | None

    # Original chunk text.
    text: str

    # Original chunk metadata.
    metadata: dict[str, Any]


# ============================================================
# Helper functions
# ============================================================

def _safe_text(value: Any) -> str:
    """
    Convert a value into safe text.

    Handles:
        - None
        - strings
        - unexpected object types
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return str(value)


def _safe_metadata(value: Any) -> dict[str, Any]:
    """
    Ensure metadata is always a dictionary.
    """

    if isinstance(value, dict):
        return value

    return {}


def _is_valid_chunk_id(chunk_id: Any) -> bool:
    """
    Return True only for non-empty string chunk IDs.
    """

    if not isinstance(chunk_id, str):
        return False

    return bool(chunk_id.strip())


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def fuse_candidates_rrf(
    dense_candidates: list[DenseCandidate] | None,
    bm25_candidates: list[BM25Candidate] | None,
    rrf_k: int = RRF_K,
) -> list[HybridCandidate]:
    """
    Combine dense and BM25 candidates using Reciprocal Rank Fusion.

    Formula:

        score(document) =
            1 / (rrf_k + dense_rank)
            +
            1 / (rrf_k + bm25_rank)

    A document appearing in only one retriever receives only that
    retriever's contribution.

    Edge cases handled:
        - dense_candidates is None
        - bm25_candidates is None
        - both candidate lists are empty
        - duplicate chunk IDs within a retriever
        - duplicate chunk IDs across retrievers
        - missing / invalid chunk IDs
        - missing text
        - missing metadata
        - invalid RRF constant
        - deterministic tie-breaking
    """

    # --------------------------------------------------------
    # Validate RRF constant
    # --------------------------------------------------------

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
        raise TypeError(
            f"rrf_k must be an integer. Received: {type(rrf_k).__name__}"
        )

    if rrf_k < 0:
        raise ValueError(
            f"rrf_k must be >= 0. Received: {rrf_k}"
        )

    # --------------------------------------------------------
    # Normalize None to empty lists
    # --------------------------------------------------------

    dense_candidates = dense_candidates or []
    bm25_candidates = bm25_candidates or []

    # --------------------------------------------------------
    # Both retrievers returned nothing
    # --------------------------------------------------------

    if not dense_candidates and not bm25_candidates:
        return []

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    rrf_scores: dict[str, float] = {}

    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    # chunk_id -> (text, metadata)
    lookup: dict[str, tuple[str, dict[str, Any]]] = {}

    # ========================================================
    # Dense retrieval contribution
    # ========================================================

    seen_dense: set[str] = set()

    for rank, candidate in enumerate(
        dense_candidates,
        start=1,
    ):

        # Defensive protection against unexpected None values.
        if candidate is None:
            continue

        chunk_id = getattr(
            candidate,
            "chunk_id",
            None,
        )

        if not _is_valid_chunk_id(chunk_id):
            continue

        chunk_id = chunk_id.strip()

        # Ignore duplicate chunk IDs inside dense results.
        #
        # Important:
        # enumerate rank remains unchanged intentionally.
        # The retriever's original ranking is preserved.
        if chunk_id in seen_dense:
            continue

        seen_dense.add(chunk_id)

        dense_ranks[chunk_id] = rank

        text = _safe_text(
            getattr(candidate, "text", "")
        )

        metadata = _safe_metadata(
            getattr(candidate, "metadata", {})
        )

        # Dense information gets first priority for storage.
        lookup[chunk_id] = (
            text,
            metadata,
        )

        contribution = 1.0 / (
            rrf_k + rank
        )

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0.0)
            + contribution
        )

    # ========================================================
    # BM25 retrieval contribution
    # ========================================================

    seen_bm25: set[str] = set()

    for rank, candidate in enumerate(
        bm25_candidates,
        start=1,
    ):

        if candidate is None:
            continue

        chunk_id = getattr(
            candidate,
            "chunk_id",
            None,
        )

        if not _is_valid_chunk_id(chunk_id):
            continue

        chunk_id = chunk_id.strip()

        # Ignore duplicates inside BM25 results.
        if chunk_id in seen_bm25:
            continue

        seen_bm25.add(chunk_id)

        bm25_ranks[chunk_id] = rank

        text = _safe_text(
            getattr(candidate, "text", "")
        )

        metadata = _safe_metadata(
            getattr(candidate, "metadata", {})
        )

        # If dense retrieval already provided this chunk,
        # keep its text/metadata unless they are empty.
        if chunk_id not in lookup:

            lookup[chunk_id] = (
                text,
                metadata,
            )

        else:

            existing_text, existing_metadata = (
                lookup[chunk_id]
            )

            if not existing_text and text:
                existing_text = text

            if not existing_metadata and metadata:
                existing_metadata = metadata

            lookup[chunk_id] = (
                existing_text,
                existing_metadata,
            )

        contribution = 1.0 / (
            rrf_k + rank
        )

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0.0)
            + contribution
        )

    # --------------------------------------------------------
    # Build final candidates
    # --------------------------------------------------------

    fused: list[HybridCandidate] = []

    for chunk_id, score in rrf_scores.items():

        text, metadata = lookup.get(
            chunk_id,
            ("", {}),
        )

        fused.append(
            HybridCandidate(
                chunk_id=chunk_id,
                rrf_score=float(score),
                dense_rank=dense_ranks.get(
                    chunk_id
                ),
                bm25_rank=bm25_ranks.get(
                    chunk_id
                ),
                text=text,
                metadata=metadata,
            )
        )

    # --------------------------------------------------------
    # Deterministic sorting
    #
    # Priority:
    #
    # 1. Higher RRF score
    # 2. Better dense rank
    # 3. Better BM25 rank
    # 4. Alphabetical chunk_id
    # --------------------------------------------------------

    def sort_key(
        candidate: HybridCandidate,
    ) -> tuple:

        dense_rank = (
            candidate.dense_rank
            if candidate.dense_rank is not None
            else float("inf")
        )

        bm25_rank = (
            candidate.bm25_rank
            if candidate.bm25_rank is not None
            else float("inf")
        )

        return (
            -candidate.rrf_score,
            dense_rank,
            bm25_rank,
            candidate.chunk_id,
        )

    fused.sort(
        key=sort_key
    )

    return fused


# ============================================================
# Hybrid search
# ============================================================

@traceable(
    name="hybrid_retrieval",
    run_type="retriever",
)
def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    hybrid_top_k: int = HYBRID_TOP_K,
) -> list[HybridCandidate]:
    """
    Run dense retrieval + BM25 retrieval and combine the results
    using Reciprocal Rank Fusion.

    Edge cases handled:
        - query is not a string
        - empty query
        - whitespace-only query
        - missing BM25 index
        - invalid top-k values
        - dense retrieval returns None
        - BM25 retrieval returns None
        - one retriever fails
        - both retrievers fail
        - fewer than requested candidates available
    """

    # --------------------------------------------------------
    # Query validation
    # --------------------------------------------------------

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string."
        )

    query = query.strip()

    if not query:
        return []

    # --------------------------------------------------------
    # BM25 index validation
    # --------------------------------------------------------

    if bm25_index is None:
        raise ValueError(
            "bm25_index cannot be None."
        )

    # --------------------------------------------------------
    # Top-k validation
    # --------------------------------------------------------

    for name, value in (
        ("dense_top_k", dense_top_k),
        ("bm25_top_k", bm25_top_k),
        ("hybrid_top_k", hybrid_top_k),
    ):

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than 0."
            )

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_candidates: list[DenseCandidate] = []

    try:

        result = dense_search(
            query,
            top_k=dense_top_k,
        )

        dense_candidates = result or []

        print(
            "[hybrid] Dense retrieval returned "
            f"{len(dense_candidates)} candidates."
        )

    except Exception as exc:

        print(
            "[hybrid] Dense retrieval failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_candidates: list[BM25Candidate] = []

    try:

        result = bm25_index.search(
            query,
            top_k=bm25_top_k,
        )

        bm25_candidates = result or []

        print(
            "[hybrid] BM25 retrieval returned "
            f"{len(bm25_candidates)} candidates."
        )

    except Exception as exc:

        print(
            "[hybrid] BM25 retrieval failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # If both retrieval systems failed / returned nothing
    # --------------------------------------------------------

    if not dense_candidates and not bm25_candidates:

        print(
            "[hybrid] No candidates returned "
            "by either retriever."
        )

        return []

    # --------------------------------------------------------
    # RRF fusion
    # --------------------------------------------------------

    fused = fuse_candidates_rrf(
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
        rrf_k=RRF_K,
    )

    print(
        "[hybrid] Unique candidates after "
        f"RRF fusion: {len(fused)}"
    )

    # --------------------------------------------------------
    # Final top-k
    # --------------------------------------------------------

    final_candidates = fused[
        :hybrid_top_k
    ]

    print(
        "[hybrid] Returning top "
        f"{len(final_candidates)} candidates."
    )

    return final_candidates


# ============================================================
# Standalone test
# ============================================================

def _run_test() -> None:
    """
    Test hybrid retrieval using the persisted BM25 index.
    """

    from src.config import BM25_INDEX_PATH

    print("=" * 60)
    print(
        "IIITDM Knowledge RAG - "
        "Hybrid Retrieval Test"
    )
    print("=" * 60)

    query = (
        "What is the hostel leave policy?"
    )

    print()
    print("[test] Query:")
    print(query)

    # --------------------------------------------------------
    # Check BM25 index exists
    # --------------------------------------------------------

    if not BM25_INDEX_PATH.exists():

        print()
        print(
            "[test] ERROR: BM25 index does not exist."
        )

        print(
            "[test] Expected path:"
        )

        print(BM25_INDEX_PATH)

        print()
        print(
            "Run:"
        )

        print(
            "python scripts/rebuild_index.py"
        )

        return

    # --------------------------------------------------------
    # Load BM25
    # --------------------------------------------------------

    print()
    print(
        "[test] Loading persisted BM25 index..."
    )

    try:

        bm25_index = BM25Index.load(
            str(BM25_INDEX_PATH)
        )

    except Exception as exc:

        print(
            "[test] ERROR loading BM25 index:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return

    print(
        "[test] BM25 index loaded successfully."
    )

    # --------------------------------------------------------
    # Run hybrid retrieval
    # --------------------------------------------------------

    print()
    print(
        "[test] Running hybrid retrieval..."
    )

    try:

        results = hybrid_search(
            query=query,
            bm25_index=bm25_index,
        )

    except Exception as exc:

        print()
        print(
            "[test] HYBRID RETRIEVAL FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()

    print(
        f"[test] Final candidates: "
        f"{len(results)}"
    )

    print()
    print("=" * 60)
    print(
        "TOP RETRIEVAL RESULTS"
    )
    print("=" * 60)

    if not results:

        print(
            "No candidates found."
        )

        return

    for i, candidate in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"[{i}]")

        print(
            f"Chunk ID: "
            f"{candidate.chunk_id}"
        )

        print(
            f"RRF score: "
            f"{candidate.rrf_score:.8f}"
        )

        print(
            f"Dense rank: "
            f"{candidate.dense_rank}"
        )

        print(
            f"BM25 rank: "
            f"{candidate.bm25_rank}"
        )

        print(
            f"Source: "
            f"{candidate.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{candidate.metadata.get('page', 'Unknown')}"
        )

        preview = (
            candidate.text[:300]
            .replace("\n", " ")
            .strip()
        )

        if not preview:
            preview = "[No text available]"

        print(
            f"Text preview: {preview}"
        )

    print()
    print("=" * 60)
    print(
        "HYBRID RETRIEVAL TEST COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    _run_test()
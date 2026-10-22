"""
reranker.py

Cross-encoder reranking for hybrid retrieval results.

Pipeline:

    Dense Retrieval
          +
        BM25
          |
          v
      RRF Fusion
          |
          v
    Top-N Candidates
          |
          v
    Cross-Encoder Reranker
          |
          v
    Final Top-K Evidence

Unlike embedding-based retrieval, where the query and document are
encoded independently, a cross-encoder processes the query and document
together. This usually provides more accurate relevance ranking, but is
computationally more expensive.

Therefore, the reranker operates only on the already narrowed hybrid
candidate set.

Default model:
    cross-encoder/ms-marco-MiniLM-L-6-v2

Features:
    - Lazy model loading
    - CUDA support when available
    - Automatic CPU fallback
    - Batch inference
    - Empty input handling
    - Invalid candidate protection
    - Duplicate chunk protection
    - Safe sigmoid conversion
    - Metadata preservation

Input:
    query
    +
    HybridCandidate list

Output:
    Ranked RerankedCandidate list
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None



from src.config import (
    RERANKER_MODEL_NAME,
    RERANK_TOP_K,
)

from src.hybrid_retrieval import HybridCandidate


# ---------------------------------------------------------
# Cached singleton
# ---------------------------------------------------------

_reranker_model: CrossEncoder | None = None
_reranker_device: str | None = None


@dataclass
class RerankedCandidate:
    """
    Candidate after cross-encoder reranking.
    """

    chunk_id: str

    # Sigmoid-normalized relevance score in [0, 1].
    relevance_score: float

    # Original cross-encoder output.
    raw_score: float

    text: str

    metadata: dict[str, Any]


def _get_device() -> str:
    """
    Select CUDA when available.

    Falls back safely to CPU.
    """

    if torch is None:
        print(
            "[reranker] PyTorch unavailable. "
            "Using CPU."
        )
        return "cpu"

    if torch.cuda.is_available():

        try:
            device_name = torch.cuda.get_device_name(0)

            print(
                f"[reranker] CUDA available: "
                f"{device_name}"
            )

            return "cuda"

        except Exception as error:

            print(
                "[reranker] CUDA detected but device "
                f"initialization failed: {error}"
            )

            print(
                "[reranker] Falling back to CPU."
            )

    print("[reranker] Using CPU.")

    return "cpu"


def get_reranker_model() -> CrossEncoder:
    """
    Load and cache the CrossEncoder model.

    CUDA is used when available.
    CPU is used as a fallback.
    """

    global _reranker_model
    global _reranker_device

    if _reranker_model is not None:
        return _reranker_model

    if CrossEncoder is None:

        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    device = _get_device()

    print("=" * 60)
    print("IIITDM Knowledge RAG - Reranker")
    print("=" * 60)

    print(
        f"[reranker] Model: "
        f"{RERANKER_MODEL_NAME}"
    )

    print(
        f"[reranker] Trying device: "
        f"{device}"
    )

    try:

        _reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            device=device,
        )

        _reranker_device = device

        print(
            "[reranker] Model loaded successfully."
        )

        print(
            f"[reranker] Device: "
            f"{_reranker_device}"
        )

        return _reranker_model

    except Exception as error:

        # -------------------------------------------------
        # GPU failed -> retry CPU.
        # -------------------------------------------------

        if device == "cuda":

            print(
                "[reranker] WARNING: CUDA model loading "
                "failed."
            )

            print(
                f"[reranker] Error: "
                f"{type(error).__name__}: {error}"
            )

            print(
                "[reranker] Retrying on CPU..."
            )

            try:

                _reranker_model = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    device="cpu",
                )

                _reranker_device = "cpu"

                print(
                    "[reranker] Model loaded successfully "
                    "on CPU."
                )

                return _reranker_model

            except Exception as cpu_error:

                raise RuntimeError(
                    "Failed to load reranker on both CUDA "
                    "and CPU."
                ) from cpu_error

        raise RuntimeError(
            "Failed to load reranker model."
        ) from error


def get_reranker_device() -> str:
    """
    Return the device currently used by the reranker.
    """

    if _reranker_model is None:
        get_reranker_model()

    return _reranker_device or "unknown"


def _safe_sigmoid(value: float) -> float:
    """
    Numerically stable sigmoid.

    Converts a raw cross-encoder score into approximately [0, 1].
    """

    value = float(value)

    if value >= 0:

        z = math.exp(-value)

        return 1.0 / (1.0 + z)

    z = math.exp(value)

    return z / (1.0 + z)


def _clean_candidates(
    candidates: list[HybridCandidate],
) -> list[HybridCandidate]:
    """
    Remove invalid and duplicate candidates.

    The first occurrence of a chunk is kept because the incoming
    hybrid list is already ranked by RRF.
    """

    cleaned: list[HybridCandidate] = []

    seen: set[str] = set()

    for candidate in candidates:

        chunk_id = candidate.chunk_id

        text = candidate.text

        if not chunk_id:
            continue

        if not isinstance(text, str):
            continue

        if not text.strip():
            continue

        if chunk_id in seen:
            continue

        seen.add(chunk_id)

        cleaned.append(candidate)

    return cleaned


def rerank(
    query: str,
    candidates: list[HybridCandidate],
    top_k: int = RERANK_TOP_K,
) -> list[RerankedCandidate]:
    """
    Rerank hybrid retrieval candidates using a cross-encoder.

    The cross-encoder receives:

        (query, chunk_text)

    for every candidate and jointly scores their relevance.

    Parameters
    ----------
    query:
        User question.

    candidates:
        Candidates returned from hybrid retrieval.

    top_k:
        Number of final candidates to return.

    Returns
    -------
    list[RerankedCandidate]
        Candidates sorted from highest to lowest relevance.
    """

    # ---------------------------------------------------------
    # Validate query.
    # ---------------------------------------------------------

    query = query.strip()

    if not query:

        raise ValueError(
            "Cannot rerank candidates for an empty query."
        )

    # ---------------------------------------------------------
    # Validate top_k.
    # ---------------------------------------------------------

    if top_k <= 0:

        raise ValueError(
            f"top_k must be positive. Got: {top_k}"
        )

    # ---------------------------------------------------------
    # Empty candidate list.
    # ---------------------------------------------------------

    if not candidates:

        print(
            "[reranker] No candidates received."
        )

        return []

    # ---------------------------------------------------------
    # Remove invalid / duplicate candidates.
    # ---------------------------------------------------------

    candidates = _clean_candidates(candidates)

    if not candidates:

        print(
            "[reranker] No valid candidates remain "
            "after cleaning."
        )

        return []

    print(
        f"[reranker] Candidates received: "
        f"{len(candidates)}"
    )

    # ---------------------------------------------------------
    # Load model.
    # ---------------------------------------------------------

    model = get_reranker_model()

    # ---------------------------------------------------------
    # Construct (query, document) pairs.
    # ---------------------------------------------------------

    pairs = [
        (query, candidate.text)
        for candidate in candidates
    ]

    # ---------------------------------------------------------
    # Run cross-encoder inference.
    # ---------------------------------------------------------

    try:

        raw_scores = model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

    except RuntimeError as error:

        error_text = str(error).lower()

        # GPU OOM fallback.
        if (
            "out of memory" in error_text
            or "cuda" in error_text
        ):

            print(
                "[reranker] WARNING: GPU inference failed."
            )

            print(
                "[reranker] Retrying with smaller "
                "batch size..."
            )

            try:

                raw_scores = model.predict(
                    pairs,
                    batch_size=4,
                    show_progress_bar=False,
                )

            except Exception as retry_error:

                raise RuntimeError(
                    "Reranker inference failed."
                ) from retry_error

        else:

            raise RuntimeError(
                "Reranker inference failed."
            ) from error

    # ---------------------------------------------------------
    # Pair candidates with scores.
    # ---------------------------------------------------------

    scored = list(
        zip(
            candidates,
            raw_scores,
        )
    )

    # Sort by raw relevance score.
    scored.sort(
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    # ---------------------------------------------------------
    # Build final results.
    # ---------------------------------------------------------

    reranked: list[RerankedCandidate] = []

    for candidate, raw_score in scored[:top_k]:

        raw_score = float(raw_score)

        relevance_score = _safe_sigmoid(
            raw_score
        )

        reranked.append(
            RerankedCandidate(

                chunk_id=candidate.chunk_id,

                relevance_score=relevance_score,

                raw_score=raw_score,

                text=candidate.text,

                metadata=dict(
                    candidate.metadata
                ),
            )
        )

    print(
        f"[reranker] Candidates after reranking: "
        f"{len(reranked)}"
    )

    return reranked


if __name__ == "__main__":
    from src.bm25 import BM25Index
    from src.config import BM25_INDEX_PATH
    from src.hybrid_retrieval import hybrid_search

    print("=" * 60)
    print("IIITDM Knowledge RAG - Reranker Test")
    print("=" * 60)

    query = "What is the hostel leave policy?"

    print("\n[test] Query:")
    print(query)

    # ---------------------------------------------------------
    # Load BM25 index
    # ---------------------------------------------------------

    print("\n[test] Loading persisted BM25 index...")

    try:
        bm25_index = BM25Index.load(BM25_INDEX_PATH)

    except FileNotFoundError:
        print("\nERROR: BM25 index was not found.")
        print("Run:")
        print("    python scripts/rebuild_index.py")
        raise SystemExit(1)

    except Exception as exc:
        print(f"\nERROR: Failed to load BM25 index: {exc}")
        raise SystemExit(1)

    print("[test] BM25 index loaded successfully.")

    # ---------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------

    print("\n[test] Running hybrid retrieval...")

    try:
        hybrid_candidates = hybrid_search(
            query=query,
            bm25_index=bm25_index,
        )

    except Exception as exc:
        print(f"\nERROR: Hybrid retrieval failed: {exc}")
        raise SystemExit(1)

    if not hybrid_candidates:
        print("\n[test] No hybrid candidates were retrieved.")
        raise SystemExit(0)

    print(
        f"[test] Hybrid candidates retrieved: "
        f"{len(hybrid_candidates)}"
    )

    # ---------------------------------------------------------
    # Reranking
    # ---------------------------------------------------------

    print("\n[test] Running cross-encoder reranking...")

    try:
        reranked_candidates = rerank(
            query=query,
            candidates=hybrid_candidates,
        )

    except Exception as exc:
        print(f"\nERROR: Reranking failed: {exc}")
        raise SystemExit(1)

    if not reranked_candidates:
        print("\n[test] No candidates remained after reranking.")
        raise SystemExit(0)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL RERANKED RESULTS")
    print("=" * 60)

    for i, candidate in enumerate(reranked_candidates, start=1):

        metadata = (
            candidate.metadata
            if isinstance(candidate.metadata, dict)
            else {}
        )

        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or "Unknown"
        )

        page = (
            metadata.get("page")
            or metadata.get("page_number")
            or "Unknown"
        )

        text = candidate.text or ""

        preview = " ".join(text.split())

        if len(preview) > 350:
            preview = preview[:350] + "..."

        print()
        print(f"[{i}]")
        print(f"Chunk ID: {candidate.chunk_id}")
        print(
            f"Relevance score: "
            f"{candidate.relevance_score:.6f}"
        )
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Text preview: {preview}")

    print()
    print("=" * 60)
    print("RERANKER TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
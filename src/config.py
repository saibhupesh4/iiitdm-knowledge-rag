"""
config.py

Centralized configuration for the IIITDM Knowledge RAG project.

Every other module should import settings from this module instead of
reading environment variables directly.

Configuration priority:

    1. Environment variables already set in the operating system
    2. Values from PROJECT_ROOT/.env
    3. Safe default values defined in this file

The .env file is loaded explicitly from the project root so that the
application works correctly regardless of the directory from which
Python is executed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# =============================================================
# Project Root
# =============================================================

# src/config.py
# parent      -> src
# parent.parent -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================
# Environment File
# =============================================================

# Always look for .env in the project root.
#
# This avoids problems caused by:
#
#     load_dotenv()
#
# searching relative to the current working directory.
ENV_PATH = PROJECT_ROOT / ".env"

# Load the environment file.
#
# override=False means values explicitly set in the operating system
# take priority over values inside .env.
load_dotenv(dotenv_path=ENV_PATH, override=False)


# =============================================================
# Helper Functions
# =============================================================

def _get_env_string(key: str, default: str = "") -> str:
    """
    Read and normalize a string environment variable.

    Whitespace is removed so values such as:

        LM_STUDIO_MODEL= model-name

    do not accidentally include leading/trailing spaces.
    """
    value = os.getenv(key, default)

    if value is None:
        return default

    return value.strip()


def _get_optional_string(key: str) -> str | None:
    """
    Return a cleaned string value or None if the value is empty.

    Also treats common placeholder values as unset.
    """
    value = os.getenv(key)

    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    placeholders = {
        "your_model_name",
        "none",
        "null",
    }

    if value.lower() in placeholders:
        return None

    return value


def _get_int(key: str, default: int) -> int:
    """
    Read an integer environment variable.

    Raises a clear error instead of silently accepting invalid values.
    """
    raw_value = os.getenv(key)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{key}' must be an integer. "
            f"Received: {raw_value!r}"
        ) from exc


def _get_float(key: str, default: float) -> float:
    """
    Read a floating-point environment variable.
    """
    raw_value = os.getenv(key)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{key}' must be a number. "
            f"Received: {raw_value!r}"
        ) from exc


def _resolve_project_path(value: str | None, default: Path) -> Path:
    """
    Convert a configured path into an absolute Path.

    Relative paths from .env are interpreted relative to PROJECT_ROOT,
    not the directory from which Python was launched.
    """
    if not value or not value.strip():
        return default.resolve()

    path = Path(value.strip())

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# =============================================================
# Project Paths
# =============================================================

DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"

STORAGE_PATH = PROJECT_ROOT / "storage"

CHROMA_PATH = _resolve_project_path(
    os.getenv("CHROMA_PATH"),
    STORAGE_PATH / "chroma",
)

BM25_INDEX_PATH = _resolve_project_path(
    os.getenv("BM25_INDEX_PATH"),
    STORAGE_PATH / "bm25_index.pkl",
)


# =============================================================
# LM Studio
# =============================================================

# OpenAI-compatible LM Studio API endpoint.
LM_STUDIO_BASE_URL: str = _get_env_string(
    "LM_STUDIO_BASE_URL",
    "http://127.0.0.1:1234/v1",
).rstrip("/")

# Optional explicit model ID.
#
# If None, src.llm.py may automatically detect an available model
# from LM Studio's /models endpoint.
LM_STUDIO_MODEL: str | None = _get_optional_string(
    "LM_STUDIO_MODEL"
)

# The OpenAI Python SDK requires a non-empty API key.
#
# This is only a local placeholder for LM Studio.
# It is NOT an OpenAI API key.
LM_STUDIO_PLACEHOLDER_KEY: str = "lm-studio"


# =============================================================
# Embedding Model
# =============================================================

EMBEDDING_MODEL_NAME: str = _get_env_string(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
)


# =============================================================
# Cross-Encoder Reranker
# =============================================================

RERANKER_MODEL_NAME: str = _get_env_string(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# =============================================================
# Chroma Vector Store
# =============================================================

CHROMA_COLLECTION_NAME: str = _get_env_string(
    "CHROMA_COLLECTION_NAME",
    "iiitdm_docs",
)


# =============================================================
# Document Chunking
# =============================================================

CHUNK_SIZE: int = _get_int(
    "CHUNK_SIZE",
    900,
)

CHUNK_OVERLAP: int = _get_int(
    "CHUNK_OVERLAP",
    125,
)


# =============================================================
# Retrieval
# =============================================================

DENSE_TOP_K: int = _get_int(
    "DENSE_TOP_K",
    20,
)

BM25_TOP_K: int = _get_int(
    "BM25_TOP_K",
    20,
)

HYBRID_TOP_K: int = _get_int(
    "HYBRID_TOP_K",
    20,
)

RERANK_TOP_K: int = _get_int(
    "RERANK_TOP_K",
    5,
)


# =============================================================
# Reciprocal Rank Fusion
# =============================================================

RRF_K: int = _get_int(
    "RRF_K",
    60,
)


# =============================================================
# Evidence Sufficiency Gate
# =============================================================

EVIDENCE_SCORE_THRESHOLD: float = _get_float(
    "EVIDENCE_SCORE_THRESHOLD",
    0.5,
)

EVIDENCE_MIN_CHUNKS: int = _get_int(
    "EVIDENCE_MIN_CHUNKS",
    2,
)


# =============================================================
# Query Reformulation
# =============================================================

MAX_QUERY_REFORMULATION_RETRIES: int = _get_int(
    "MAX_QUERY_REFORMULATION_RETRIES",
    1,
)


# =============================================================
# Query Categories
# =============================================================

QUERY_CATEGORIES: list[str] = [
    "academic",
    "hostel",
    "placement",
    "institute",
    "unknown",
]



# =============================================================
# Configuration Validation
# =============================================================

def validate_config() -> None:
    """
    Validate relationships between configuration values.

    This catches invalid configurations early instead of allowing
    failures later inside retrieval or reranking.
    """

    # -------------------------
    # Chunking validation
    # -------------------------

    if CHUNK_SIZE <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than 0."
        )

    if CHUNK_OVERLAP < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative."
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    # -------------------------
    # Retrieval validation
    # -------------------------

    if DENSE_TOP_K <= 0:
        raise ValueError(
            "DENSE_TOP_K must be greater than 0."
        )

    if BM25_TOP_K <= 0:
        raise ValueError(
            "BM25_TOP_K must be greater than 0."
        )

    if HYBRID_TOP_K <= 0:
        raise ValueError(
            "HYBRID_TOP_K must be greater than 0."
        )

    if RERANK_TOP_K <= 0:
        raise ValueError(
            "RERANK_TOP_K must be greater than 0."
        )

    # The reranker cannot return more candidates than it receives.
    if RERANK_TOP_K > HYBRID_TOP_K:
        raise ValueError(
            "RERANK_TOP_K cannot be greater than HYBRID_TOP_K."
        )

    # -------------------------
    # RRF validation
    # -------------------------

    if RRF_K <= 0:
        raise ValueError(
            "RRF_K must be greater than 0."
        )

    # -------------------------
    # Evidence gate validation
    # -------------------------

    if not 0.0 <= EVIDENCE_SCORE_THRESHOLD <= 1.0:
        raise ValueError(
            "EVIDENCE_SCORE_THRESHOLD must be between 0 and 1."
        )

    if EVIDENCE_MIN_CHUNKS <= 0:
        raise ValueError(
            "EVIDENCE_MIN_CHUNKS must be greater than 0."
        )

    # It is impossible to require more qualifying chunks than
    # the reranker returns.
    if EVIDENCE_MIN_CHUNKS > RERANK_TOP_K:
        raise ValueError(
            "EVIDENCE_MIN_CHUNKS cannot be greater than RERANK_TOP_K."
        )

    # -------------------------
    # Query reformulation validation
    # -------------------------

    if MAX_QUERY_REFORMULATION_RETRIES < 0:
        raise ValueError(
            "MAX_QUERY_REFORMULATION_RETRIES cannot be negative."
        )

    # -------------------------
    # LM Studio validation
    # -------------------------

    if not LM_STUDIO_BASE_URL:
        raise ValueError(
            "LM_STUDIO_BASE_URL cannot be empty."
        )


# =============================================================
# Run Validation on Import
# =============================================================

# Fail early if configuration is invalid.
validate_config()


# =============================================================
# Standalone Configuration Test
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IIITDM Knowledge RAG - Configuration")
    print("=" * 60)

    print()
    print("Project root:", PROJECT_ROOT)

    print()
    print(".env path:", ENV_PATH)
    print(".env exists:", ENV_PATH.exists())

    print()
    print("Raw data path:", RAW_DATA_PATH)
    print("Chroma path:", CHROMA_PATH)
    print("BM25 index path:", BM25_INDEX_PATH)

    print()
    print("LM Studio base URL:", LM_STUDIO_BASE_URL)
    print(
        "LM Studio model:",
        LM_STUDIO_MODEL
        if LM_STUDIO_MODEL
        else "Auto-detect from LM Studio",
    )

    print()
    print("Embedding model:", EMBEDDING_MODEL_NAME)
    print("Reranker model:", RERANKER_MODEL_NAME)

    print()
    print("Dense top-k:", DENSE_TOP_K)
    print("BM25 top-k:", BM25_TOP_K)
    print("Hybrid top-k:", HYBRID_TOP_K)
    print("Rerank top-k:", RERANK_TOP_K)

    print()
    print("RRF k:", RRF_K)

    print()
    print("Evidence threshold:", EVIDENCE_SCORE_THRESHOLD)
    print("Minimum evidence chunks:", EVIDENCE_MIN_CHUNKS)

    print()
    print(
        "Maximum reformulation retries:",
        MAX_QUERY_REFORMULATION_RETRIES,
    )

    print()

    print()
    print("Configuration loaded successfully.")

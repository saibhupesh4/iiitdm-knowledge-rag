"""
embeddings.py

STEP 4 of the pipeline: BGE-SMALL EMBEDDINGS.

Responsible for:
    - Loading the embedding model
    - Embedding document chunks during indexing
    - Embedding user queries during retrieval
    - Automatically using CUDA when available
    - Falling back to CPU if GPU loading fails

Embedding model:
    BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

from typing import Literal

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME


# Cached embedding model.
# Loaded only once per Python process.
_embedding_model: HuggingFaceEmbeddings | None = None

# Tracks the device actually used.
_embedding_device: Literal["cuda", "cpu"] | None = None


def _get_preferred_device() -> Literal["cuda", "cpu"]:
    """
    Return CUDA when available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def _create_embedding_model(
    device: Literal["cuda", "cpu"],
) -> HuggingFaceEmbeddings:
    """
    Create an embedding model on the requested device.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the embedding model.

    Device strategy:
        1. Try CUDA if available.
        2. If CUDA model loading fails, automatically retry on CPU.
        3. Cache the successfully loaded model.
    """
    global _embedding_model
    global _embedding_device

    if _embedding_model is not None:
        return _embedding_model

    print("=" * 60)
    print("IIITDM Knowledge RAG - Embedding Model")
    print("=" * 60)
    print(f"[embeddings] Model: {EMBEDDING_MODEL_NAME}")

    preferred_device = _get_preferred_device()

    if preferred_device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[embeddings] CUDA available: {gpu_name}")
    else:
        print("[embeddings] CUDA unavailable. Using CPU.")

    # First attempt: preferred device.
    try:
        print(
            f"[embeddings] Trying LangChain backend "
            f"on {preferred_device}..."
        )

        _embedding_model = _create_embedding_model(
            preferred_device
        )

        _embedding_device = preferred_device

        print("[embeddings] Model loaded successfully.")
        print("[embeddings] Backend: langchain_huggingface")
        print(f"[embeddings] Device: {_embedding_device}")

        return _embedding_model

    except Exception as exc:
        # CUDA failed -> retry with CPU.
        if preferred_device == "cuda":
            print(
                "[embeddings] WARNING: GPU model loading failed."
            )
            print(
                f"[embeddings] Reason: {type(exc).__name__}: {exc}"
            )
            print(
                "[embeddings] Falling back to CPU..."
            )

            try:
                # Clear partially allocated GPU memory.
                torch.cuda.empty_cache()

                _embedding_model = _create_embedding_model(
                    "cpu"
                )

                _embedding_device = "cpu"

                print(
                    "[embeddings] CPU fallback successful."
                )
                print(
                    "[embeddings] Backend: langchain_huggingface"
                )
                print(
                    f"[embeddings] Device: {_embedding_device}"
                )

                return _embedding_model

            except Exception as cpu_exc:
                raise RuntimeError(
                    "Failed to load embedding model on both "
                    "CUDA and CPU."
                ) from cpu_exc

        # CPU was already the preferred device.
        raise RuntimeError(
            "Failed to load embedding model on CPU."
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of document chunks.

    Used during indexing.
    """
    if not texts:
        return []

    model = get_embedding_model()

    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Embed a single user query.

    Used during dense retrieval.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "Cannot embed an empty query."
        )

    model = get_embedding_model()

    return model.embed_query(query)


def get_embedding_info() -> dict[str, str]:
    """
    Return information about the currently loaded embedding backend.
    """
    return {
        "model": EMBEDDING_MODEL_NAME,
        "backend": "langchain_huggingface",
        "device": _embedding_device or "not_loaded",
    }


if __name__ == "__main__":

    test_query = "What is the hostel leave policy?"

    print("\n[test] Query:")
    print(test_query)

    print("\n[test] Testing query embedding...")

    vector = embed_query(test_query)

    print(
        f"[test] Query embedding dimension: "
        f"{len(vector)}"
    )

    print("[test] First 10 values:")
    print(vector[:10])

    print("\n[test] Testing document embeddings...")

    test_documents = [
        "Students must follow the hostel rules.",
        "Leave permission must be obtained before leaving.",
        "The academic calendar contains semester dates.",
    ]

    document_vectors = embed_texts(test_documents)

    print(
        f"[test] Documents embedded: "
        f"{len(document_vectors)}"
    )

    if document_vectors:
        print(
            f"[test] Document embedding dimension: "
            f"{len(document_vectors[0])}"
        )

    print("\n" + "=" * 60)
    print("EMBEDDING TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nModel information:")

    for key, value in get_embedding_info().items():
        print(f"  {key}: {value}")
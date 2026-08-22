"""
scripts/rebuild_index.py

Full indexing entry point for the IIITDM Knowledge RAG system.

Pipeline:

    PDF files
        ↓
    ingestion
        ↓
    cleaning
        ↓
    chunking
        ↓
    ┌─────────────────┐
    │                 │
    Chroma           BM25
    │                 │
    └────────┬────────┘
             ↓
       Persisted indexes

Run this whenever documents under data/raw/ change.

The Streamlit application should only load the persisted indexes and
must not rebuild them during normal startup.

Run with:
    python scripts/rebuild_index.py
"""

from __future__ import annotations

import os
import sys


# -------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -------------------------------------------------------------
# PIPELINE IMPORTS
# -------------------------------------------------------------

from src.ingestion import load_all_documents
from src.cleaning import clean_documents
from src.chunking import chunk_documents

from src.dense_retrieval import (
    reset_vectorstore,
    index_chunks,
)

from src.bm25 import BM25Index

from src.config import (
    CHROMA_PATH,
    BM25_INDEX_PATH,
)


def main() -> None:
    """
    Run the complete document indexing pipeline.
    """

    print("=" * 60)
    print("IIITDM Knowledge RAG - Index Rebuild")
    print("=" * 60)

    # =========================================================
    # STEP 1 — INGESTION
    # =========================================================

    print("\n[1/5] Loading PDF documents...")

    raw_docs = load_all_documents()

    print(
        f"[rebuild] Raw pages loaded: "
        f"{len(raw_docs)}"
    )

    if not raw_docs:
        print(
            "[rebuild] ERROR: No PDF pages were loaded."
        )
        print(
            "[rebuild] Check data/raw/<category>/ folders."
        )
        return


    # =========================================================
    # STEP 2 — CLEANING
    # =========================================================

    print("\n[2/5] Cleaning documents...")

    cleaned_docs = clean_documents(raw_docs)

    print(
        f"[rebuild] Cleaned pages: "
        f"{len(cleaned_docs)}"
    )

    if not cleaned_docs:
        print(
            "[rebuild] ERROR: No documents remain "
            "after cleaning."
        )
        return


    # =========================================================
    # STEP 3 — CHUNKING
    # =========================================================

    print("\n[3/5] Chunking documents...")

    chunks = chunk_documents(cleaned_docs)

    print(
        f"[rebuild] Chunks produced: "
        f"{len(chunks)}"
    )

    if not chunks:
        print(
            "[rebuild] ERROR: No chunks were produced."
        )
        return


    # =========================================================
    # STEP 4 — CHROMA DENSE INDEX
    # =========================================================

    print("\n[4/5] Rebuilding Chroma vector index...")

    print(
        f"[rebuild] Chroma path: "
        f"{CHROMA_PATH}"
    )

    # Important:
    # Remove the old collection first so chunks from deleted,
    # renamed, or modified PDFs do not remain in the database.
    reset_vectorstore()

    index_chunks(chunks)


    # =========================================================
    # STEP 5 — BM25 INDEX
    # =========================================================

    print("\n[5/5] Building BM25 lexical index...")

    bm25_index = BM25Index()

    bm25_index.build(chunks)

    bm25_index.save(BM25_INDEX_PATH)

    print(
        f"[rebuild] BM25 index saved to: "
        f"{BM25_INDEX_PATH}"
    )


    # =========================================================
    # COMPLETE
    # =========================================================

    print("\n" + "=" * 60)

    print(
        "INDEX REBUILD COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print(
        f"\nTotal chunks indexed: {len(chunks)}"
    )

    print(
        f"Chroma location: {CHROMA_PATH}"
    )

    print(
        f"BM25 location: {BM25_INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
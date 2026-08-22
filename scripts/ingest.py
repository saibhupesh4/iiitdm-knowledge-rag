"""
scripts/ingest.py

Standalone CLI script for testing the document ingestion pipeline.

Pipeline:

    data/raw/
        ↓
    load_all_documents()
        ↓
    clean_documents()
        ↓
    chunk_documents()

This script does NOT build or modify:

    - Vector store
    - Dense index
    - BM25 index

Use this script to verify that PDFs are being correctly loaded,
cleaned, and chunked before running:

    python scripts/rebuild_index.py

Run:

    python scripts/ingest.py
"""

from __future__ import annotations

import os
import sys


# =============================================================
# Add project root to Python path
# =============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================
# Project imports
# =============================================================

from src.ingestion import load_all_documents
from src.cleaning import clean_documents
from src.chunking import chunk_documents


# =============================================================
# Main
# =============================================================

def main() -> None:
    """
    Run the ingestion pipeline:

        PDF loading
            ↓
        cleaning
            ↓
        chunking

    Prints summary information for each stage.
    """

    print("=" * 60)
    print("IIITDM Knowledge RAG - Ingestion Test")
    print("=" * 60)

    try:

        # -----------------------------------------------------
        # Stage 1: Load documents
        # -----------------------------------------------------

        print()
        print("[1/3] Loading PDFs from data/raw/ ...")

        raw_docs = load_all_documents()

        print(
            f"[ingestion] Loaded {len(raw_docs)} raw document pages."
        )

        if not raw_docs:
            print()
            print("[WARNING] No documents were loaded.")
            print(
                "Add PDF files under data/raw/<category>/ "
                "and run the script again."
            )
            return

        # -----------------------------------------------------
        # Stage 2: Clean documents
        # -----------------------------------------------------

        print()
        print("[2/3] Cleaning documents ...")

        cleaned_docs = clean_documents(raw_docs)

        print(
            f"[cleaning] {len(cleaned_docs)} pages remain "
            "after cleaning."
        )

        if not cleaned_docs:
            print()
            print(
                "[WARNING] No usable documents remain after cleaning."
            )
            return

        # -----------------------------------------------------
        # Stage 3: Chunk documents
        # -----------------------------------------------------

        print()
        print("[3/3] Chunking documents ...")

        chunks = chunk_documents(cleaned_docs)

        print(
            f"[chunking] Produced {len(chunks)} chunks."
        )

        # -----------------------------------------------------
        # Show example
        # -----------------------------------------------------

        if not chunks:

            print()
            print("[WARNING] No chunks were produced.")

            return

        example_chunk = chunks[0]

        print()
        print("=" * 60)
        print("EXAMPLE CHUNK")
        print("=" * 60)

        print()
        print("Metadata:")
        print(example_chunk.metadata)

        print()
        print("Text:")
        print("-" * 60)

        preview_length = 500
        preview = example_chunk.text[:preview_length]

        print(preview)

        if len(example_chunk.text) > preview_length:
            print("\n... [truncated]")

        print("-" * 60)

        # -----------------------------------------------------
        # Success summary
        # -----------------------------------------------------

        print()
        print("=" * 60)
        print("INGESTION TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as exc:

        print()
        print("=" * 60)
        print("INGESTION TEST FAILED")
        print("=" * 60)

        print()
        print(f"Error: {exc}")

        sys.exit(1)


# =============================================================
# Entry point
# =============================================================

if __name__ == "__main__":
    main()
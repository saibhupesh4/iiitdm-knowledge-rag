"""
chunking.py

STEP 3 of the pipeline:

CLEANED PAGE-LEVEL DOCUMENTS
        ↓
STRUCTURE-AWARE RECURSIVE CHUNKING
        ↓
CHUNK-LEVEL DOCUMENTS

Responsible for:
    - Splitting cleaned page-level Documents into smaller chunks
    - Using recursive splitting to preserve natural text structure
    - Preserving metadata from the original page
    - Adding a unique chunk_id to every chunk

Input:
    List of cleaned LangChain Document objects, one per PDF page.

Output:
    List of chunk-level Document objects.

Each chunk preserves:

    {
        "source": "...",
        "page": 3,
        "category": "hostel",
        "title": "...",
        "section": "Hostel Leave Rules",
        "chunk_id": "hostel_rules_2025_26_p3_c0"
    }
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# =============================================================
# Splitter construction
# =============================================================

def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Build the RecursiveCharacterTextSplitter.

    The separator order preserves structure as much as possible:

        1. Paragraph boundaries
        2. Line boundaries
        3. Sentence boundaries
        4. Word boundaries
        5. Character-level fallback
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",   # Paragraph
            "\n",     # Line
            ". ",     # Sentence
            " ",      # Word
            "",       # Final fallback
        ],

        length_function=len,
        is_separator_regex=False,
    )


# =============================================================
# Chunk ID creation
# =============================================================

def create_chunk_id(
    source: str,
    page: int,
    chunk_index: int,
) -> str:
    """
    Create a readable and stable chunk ID.

    Example:

        hostel_rules_2025_26.pdf
        page = 3
        chunk_index = 0

        ->
        hostel_rules_2025_26_p3_c0
    """

    source_name = source.rsplit(
        ".",
        1,
    )[0]

    return (
        f"{source_name}"
        f"_p{page}"
        f"_c{chunk_index}"
    )


# =============================================================
# Main chunking function
# =============================================================

def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split cleaned page-level Documents into chunk-level Documents.

    Metadata is copied from the original page to every chunk.

    Additional metadata:

        chunk_id
        chunk_index
        chunk_size_chars
    """

    splitter = build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []

    skipped_empty = 0

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "unknown",
        )

        page = doc.metadata.get(
            "page",
            0,
        )

        pieces = splitter.split_text(
            doc.page_content
        )

        for chunk_index, piece in enumerate(
            pieces
        ):

            # Safety check
            if not piece.strip():

                skipped_empty += 1

                continue

            # Preserve all existing metadata
            metadata = dict(
                doc.metadata
            )

            # Add chunk-specific metadata
            metadata["chunk_id"] = (
                create_chunk_id(
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                )
            )

            metadata["chunk_index"] = (
                chunk_index
            )

            metadata["chunk_size_chars"] = (
                len(piece)
            )

            chunks.append(
                Document(
                    page_content=piece,
                    metadata=metadata,
                )
            )

    print(
        f"[chunking] "
        f"Input pages: {len(documents)}"
    )

    print(
        f"[chunking] "
        f"Chunks produced: {len(chunks)}"
    )

    print(
        f"[chunking] "
        f"Empty chunks skipped: {skipped_empty}"
    )

    return chunks


# =============================================================
# Standalone test
# =============================================================

if __name__ == "__main__":

    from src.ingestion import (
        load_all_documents,
    )

    from src.cleaning import (
        clean_documents,
    )

    print(
        "IIITDM Knowledge RAG - Chunking Test"
    )

    print("=" * 60)

    print(
        f"\nChunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    print("\nLoading PDFs...\n")

    raw_documents = load_all_documents()

    print("\nCleaning documents...\n")

    cleaned_documents = clean_documents(
        raw_documents
    )

    print("\nChunking documents...\n")

    chunks = chunk_documents(
        cleaned_documents
    )

    print()

    print("=" * 60)

    print("\nFINAL RESULT")

    print(
        f"Raw pages: "
        f"{len(raw_documents)}"
    )

    print(
        f"Cleaned pages: "
        f"{len(cleaned_documents)}"
    )

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    if chunks:

        chunk_sizes = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        average_size = (
            sum(chunk_sizes)
            / len(chunk_sizes)
        )

        print()

        print(
            f"Smallest chunk: "
            f"{min(chunk_sizes)} characters"
        )

        print(
            f"Largest chunk: "
            f"{max(chunk_sizes)} characters"
        )

        print(
            f"Average chunk size: "
            f"{average_size:.1f} characters"
        )

        print()

        print(
            "Example chunk metadata:"
        )

        print(
            chunks[0].metadata
        )

        print()

        print(
            "Example chunk text:"
        )

        print("-" * 60)

        print(
            chunks[0].page_content
        )

        print("-" * 60)

    else:

        print(
            "\nNo chunks produced."
        )
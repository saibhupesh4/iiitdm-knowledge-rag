"""
dense_retrieval.py

Semantic retrieval against the local Chroma vector store.

Responsibilities:
    - Build and maintain the Chroma vector index
    - Perform dense semantic retrieval
    - Preserve chunk metadata

Input:
    User query (str)

Output:
    Ranked DenseCandidate objects containing:
        chunk_id
        similarity score
        chunk text
        metadata

Storage:
    storage/chroma/

The vector database is built by scripts/rebuild_index.py and loaded
during normal application execution.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    Chroma = None

from src.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION_NAME,
    DENSE_TOP_K,
)
from src.embeddings import get_embedding_model


@dataclass
class DenseCandidate:
    """
    A candidate returned by dense semantic retrieval.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict


_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """
    Return the cached Chroma vector store.

    The same embedding model is used for:
        - document indexing
        - query embedding
    """

    global _vectorstore

    if _vectorstore is None:

        if Chroma is None:
            raise ImportError(
                "langchain_chroma is not installed. "
                "Run: pip install -r requirements.txt"
            )

        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=CHROMA_PATH,
        )

    return _vectorstore


def _sanitize_metadata(metadata: dict) -> dict:
    """
    Prepare metadata for Chroma.

    Chroma metadata values must be primitive values. In particular,
    None values are removed.
    """

    clean_metadata = {}

    for key, value in metadata.items():

        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value

        else:
            clean_metadata[key] = str(value)

    return clean_metadata


def reset_vectorstore() -> None:
    """
    Delete the current Chroma collection.

    Used before a full rebuild so removed or changed chunks do not
    remain in the vector database.
    """

    global _vectorstore

    if Chroma is None:
        raise ImportError(
            "langchain_chroma is not installed. "
            "Run: pip install -r requirements.txt"
        )

    print(
        f"[dense_retrieval] Resetting collection: "
        f"{CHROMA_COLLECTION_NAME}"
    )

    try:
        vectorstore = get_vectorstore()
        vectorstore.delete_collection()

    except Exception as exc:
        print(
            "[dense_retrieval] No existing collection to reset "
            f"or reset skipped: {exc}"
        )

    _vectorstore = None


def index_chunks(
    chunks: list,
    batch_size: int = 64,
) -> None:
    """
    Add chunk-level Documents to Chroma.

    Parameters
    ----------
    chunks : list
        Chunk-level LangChain Documents.

    batch_size : int
        Number of chunks added in one batch.
    """

    if not chunks:
        print("[dense_retrieval] No chunks to index.")
        return

    vectorstore = get_vectorstore()

    total = len(chunks)

    print(
        f"[dense_retrieval] Indexing {total} chunks..."
    )

    for start in range(0, total, batch_size):

        batch = chunks[start:start + batch_size]

        ids = []
        documents = []

        for chunk in batch:

            metadata = _sanitize_metadata(
                dict(chunk.metadata)
            )

            # Update metadata with Chroma-safe values.
            chunk.metadata = metadata

            chunk_id = metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Chunk is missing required metadata field: "
                    "'chunk_id'"
                )

            ids.append(chunk_id)
            documents.append(chunk)

        vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

        end = min(start + batch_size, total)

        print(
            f"[dense_retrieval] Indexed "
            f"{end}/{total} chunks"
        )

    print(
        "[dense_retrieval] Indexing completed successfully."
    )


def dense_search(
    query: str,
    top_k: int = DENSE_TOP_K,
) -> list[DenseCandidate]:
    """
    Perform semantic similarity search.

    Parameters
    ----------
    query : str
        User's question.

    top_k : int
        Maximum number of candidates to retrieve.

    Returns
    -------
    list[DenseCandidate]
        Candidates ranked from most to least similar.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "Cannot perform dense search with an empty query."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k,
    )

    candidates: list[DenseCandidate] = []

    for doc, distance in results:

        chunk_id = doc.metadata.get(
            "chunk_id",
            "",
        )

        # Chroma distance:
        # lower distance = more similar.
        #
        # Convert to a monotonic similarity-style score:
        # higher score = more similar.
        similarity = 1.0 / (1.0 + max(float(distance), 0.0))

        candidates.append(
            DenseCandidate(
                chunk_id=chunk_id,
                score=similarity,
                text=doc.page_content,
                metadata=dict(doc.metadata),
            )
        )

    return candidates


if __name__ == "__main__":

    print("=" * 60)
    print("Dense Retrieval Test")
    print("=" * 60)

    test_query = "What is the hostel leave policy?"

    print(f"\nQuery:\n{test_query}\n")

    results = dense_search(test_query)

    print(f"Results returned: {len(results)}\n")

    for i, result in enumerate(results[:5], start=1):

        print(f"[{i}] Score: {result.score:.4f}")

        print(
            f"Source: "
            f"{result.metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{result.metadata.get('page')}"
        )

        print(
            f"Text: "
            f"{result.text[:200]}"
        )

        print("-" * 60)
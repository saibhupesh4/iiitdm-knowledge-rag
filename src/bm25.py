"""
bm25.py

STEP 6: BM25 LEXICAL RETRIEVAL.

Builds and searches a BM25 index over the same chunk corpus used by
the Chroma dense vector index.

BM25 is useful for exact lexical matching, including:

    - Course codes
    - Dates
    - Percentages
    - Policy names
    - Form names
    - Abbreviations
    - Exact terminology

The implementation is designed to be robust against malformed or
symbol-only chunks. Chunks that contain no valid BM25 tokens are skipped
instead of crashing the entire indexing pipeline.
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

from src.config import BM25_TOP_K


# Matches lowercase words and numbers.
#
# Examples:
#   "CS101"      -> cs101
#   "2025-26"    -> 2025, 26
#   "75%"        -> 75
#   "hostel"     -> hostel
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class BM25Candidate:
    """
    A chunk returned by BM25 retrieval.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


def _tokenize(text: str) -> list[str]:
    """
    Convert text into BM25 tokens.

    Strategy:
        1. Handle non-string input safely.
        2. Convert to lowercase.
        3. Extract alphanumeric tokens.

    Returns an empty list if the text contains no usable tokens.
    """

    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    BM25 index wrapper.

    Keeps the following data aligned:

        BM25 corpus
        chunk_ids
        chunk texts
        chunk metadata

    Invalid chunks that contain zero BM25 tokens are skipped during
    index construction.
    """

    def __init__(self) -> None:

        self._bm25: BM25Okapi | None = None

        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    def build(self, chunks: list[Any]) -> None:
        """
        Build the BM25 index from chunk Documents.

        Parameters
        ----------
        chunks : list
            Chunk-level LangChain Documents.

        Notes
        -----
        A chunk that produces zero lexical tokens is skipped.

        This prevents one malformed or symbol-only chunk from causing
        the entire RAG indexing pipeline to fail.
        """

        if BM25Okapi is None:
            raise ImportError(
                "rank_bm25 is not installed.\n"
                "Install it using:\n"
                "pip install rank-bm25"
            )

        if not chunks:
            raise ValueError(
                "Cannot build BM25 index from an empty chunk list."
            )

        print(
            f"[bm25] Building index from {len(chunks)} chunks..."
        )

        valid_tokenized_corpus: list[list[str]] = []

        valid_chunk_ids: list[str] = []
        valid_texts: list[str] = []
        valid_metadatas: list[dict[str, Any]] = []

        skipped_empty_text = 0
        skipped_no_tokens = 0

        for chunk_index, chunk in enumerate(chunks):

            # --------------------------------------------------
            # Safely obtain chunk text
            # --------------------------------------------------

            text = getattr(chunk, "page_content", "")

            if not isinstance(text, str):
                skipped_empty_text += 1
                continue

            text = text.strip()

            if not text:
                skipped_empty_text += 1
                continue

            # --------------------------------------------------
            # Tokenize
            # --------------------------------------------------

            tokens = _tokenize(text)

            if not tokens:
                skipped_no_tokens += 1

                print(
                    "[bm25] Warning: skipping chunk with "
                    f"no lexical tokens (index={chunk_index})"
                )

                continue

            # --------------------------------------------------
            # Safely obtain metadata
            # --------------------------------------------------

            metadata = getattr(chunk, "metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            metadata = dict(metadata)

            chunk_id = metadata.get(
                "chunk_id",
                f"bm25_chunk_{chunk_index}"
            )

            # --------------------------------------------------
            # Store aligned data
            # --------------------------------------------------

            valid_tokenized_corpus.append(tokens)

            valid_chunk_ids.append(str(chunk_id))

            valid_texts.append(text)

            valid_metadatas.append(metadata)

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        if not valid_tokenized_corpus:

            raise ValueError(
                "BM25 index could not be built because no valid "
                "chunks contained lexical tokens."
            )

        # ------------------------------------------------------
        # Build BM25
        # ------------------------------------------------------

        self._bm25 = BM25Okapi(
            valid_tokenized_corpus
        )

        self._chunk_ids = valid_chunk_ids
        self._texts = valid_texts
        self._metadatas = valid_metadatas

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        print(
            f"[bm25] Valid chunks indexed: "
            f"{len(valid_tokenized_corpus)}"
        )

        print(
            f"[bm25] Empty chunks skipped: "
            f"{skipped_empty_text}"
        )

        print(
            f"[bm25] Chunks with no BM25 tokens skipped: "
            f"{skipped_no_tokens}"
        )

        print(
            "[bm25] BM25 index built successfully."
        )

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> list[BM25Candidate]:
        """
        Search the BM25 index.

        Parameters
        ----------
        query : str
            User query.

        top_k : int
            Maximum number of candidates to return.

        Returns
        -------
        list[BM25Candidate]
        """

        if self._bm25 is None:

            raise RuntimeError(
                "BM25Index has not been built or loaded."
            )

        if not isinstance(query, str):

            raise TypeError(
                "BM25 query must be a string."
            )

        query = query.strip()

        if not query:
            return []

        tokenized_query = _tokenize(query)

        # A query containing only symbols should simply return nothing.
        if not tokenized_query:

            print(
                "[bm25] Warning: query produced no valid tokens."
            )

            return []

        scores = self._bm25.get_scores(
            tokenized_query
        )

        if len(scores) == 0:
            return []

        # Ensure top_k is valid.
        top_k = max(
            1,
            min(top_k, len(scores))
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        candidates: list[BM25Candidate] = []

        for i in ranked_indices:

            candidates.append(
                BM25Candidate(
                    chunk_id=self._chunk_ids[i],
                    score=float(scores[i]),
                    text=self._texts[i],
                    metadata=self._metadatas[i],
                )
            )

        return candidates

    def save(self, path: str) -> None:
        """
        Persist the BM25 index to disk.
        """

        if self._bm25 is None:

            raise RuntimeError(
                "Cannot save BM25 index before building it."
            )

        directory = os.path.dirname(path)

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        state = {
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }

        with open(
            path,
            "wb",
        ) as file:

            pickle.dump(
                state,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(
            f"[bm25] Index saved to: {path}"
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "BM25Index":
        """
        Load a previously saved BM25 index.
        """

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"BM25 index not found: {path}"
            )

        with open(
            path,
            "rb",
        ) as file:

            state = pickle.load(file)

        required_keys = {
            "bm25",
            "chunk_ids",
            "texts",
            "metadatas",
        }

        missing_keys = required_keys - set(state.keys())

        if missing_keys:

            raise ValueError(
                "Invalid BM25 index file. "
                f"Missing keys: {sorted(missing_keys)}"
            )

        index = cls()

        index._bm25 = state["bm25"]

        index._chunk_ids = state["chunk_ids"]

        index._texts = state["texts"]

        index._metadatas = state["metadatas"]

        print(
            f"[bm25] Loaded index with "
            f"{len(index._chunk_ids)} chunks."
        )

        return index


if __name__ == "__main__":

    print("=" * 60)

    print(
        "IIITDM Knowledge RAG - BM25"
    )

    print("=" * 60)

    print(
        "\nRun the following command to build the BM25 index:\n"
    )

    print(
        "python scripts/rebuild_index.py"
    )
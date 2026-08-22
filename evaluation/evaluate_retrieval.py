"""
evaluation/evaluate_retrieval.py

Measures retrieval quality against evaluation/questions.json using:
    Hit@5    - fraction of questions where the expected source/page
               appears anywhere in the top-5 retrieved chunks
    Recall@5 - for this project (one relevant chunk per question), this
               reduces to the same signal as Hit@5; kept as a distinct
               metric name for clarity in the ablation report and in case
               multiple relevant chunks per question are added later
    MRR      - mean reciprocal rank of the first chunk that matches the
               expected source/page, across all questions

Requires a runnable retrieval pipeline (hybrid_retrieval.py / dense_
retrieval.py / bm25.py) and a populated Chroma + BM25 index — this is an
evaluation script, not a component to run before those exist.

Run with:
    python evaluation/evaluate_retrieval.py

Input:  evaluation/questions.json
Output: printed metrics; does not write results.csv on its own (see
        evaluation/ablation.py, which calls this across pipeline variants)
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_questions(path: str = "evaluation/questions.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data["questions"] if q.get("is_answerable", True)]


def _matches(metadata: dict, expected_source: str, expected_page: int | None) -> bool:
    if metadata.get("source") != expected_source:
        return False
    if expected_page is None:
        return True
    return metadata.get("page") == expected_page


def hit_at_k(retrieved_metadatas: list[dict], expected_source: str, expected_page: int | None, k: int = 5) -> bool:
    """
    Return True if any of the top-k retrieved chunks match the expected
    (source, page).
    """
    return any(_matches(m, expected_source, expected_page) for m in retrieved_metadatas[:k])


def reciprocal_rank(retrieved_metadatas: list[dict], expected_source: str, expected_page: int | None) -> float:
    """
    Return 1 / rank of the first matching chunk (1-indexed), or 0.0 if no
    match is found in the retrieved list.
    """
    for i, m in enumerate(retrieved_metadatas, start=1):
        if _matches(m, expected_source, expected_page):
            return 1.0 / i
    return 0.0


def evaluate_retrieval(retrieval_fn, questions: list[dict] | None = None) -> dict:
    """
    Run retrieval_fn(question_text) -> list of chunk metadata dicts for
    every question, and compute Hit@5 / Recall@5 / MRR.

    retrieval_fn is injected so this same evaluation logic can be reused
    across the ablation variants (dense-only, BM25-only, hybrid, etc.) in
    ablation.py.
    """
    if questions is None:
        questions = load_questions()

    if not questions:
        return {"hit_at_5": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "num_questions": 0}

    hits = []
    rrs = []

    for q in questions:
        retrieved_metadatas = retrieval_fn(q["question"])
        expected_source = q.get("expected_source")
        expected_page = q.get("expected_page")

        if not expected_source:
            continue

        hits.append(hit_at_k(retrieved_metadatas, expected_source, expected_page, k=5))
        rrs.append(reciprocal_rank(retrieved_metadatas, expected_source, expected_page))

    n = len(hits) or 1
    return {
        "hit_at_5": sum(hits) / n,
        "recall_at_5": sum(hits) / n,
        "mrr": sum(rrs) / n,
        "num_questions": len(hits),
    }


if __name__ == "__main__":
    from src.hybrid_retrieval import hybrid_search
    from src.bm25 import BM25Index
    from src.config import BM25_INDEX_PATH

    bm25_index = BM25Index.load(BM25_INDEX_PATH)
    results = evaluate_retrieval(
        lambda q: [c.metadata for c in hybrid_search(q, bm25_index)]
    )
    print(results)

"""
evaluation/ablation.py

Runs evaluation/questions.json through six progressively more complete
pipeline configurations and compares them, per project step 27:

    A. Dense Retrieval only
    B. BM25 only
    C. Hybrid Retrieval (dense + BM25 fusion)
    D. Hybrid + Cross-Encoder Reranking
    E. Hybrid + Reranking + Evidence Sufficiency Gate
    F. Full Pipeline (adds query reformulation + claim verification)

For each variant, reuses evaluate_retrieval.py / evaluate_generation.py /
evaluate_grounding.py where applicable (not every metric applies to every
variant — e.g. A/B/C have no reranker score, D/E/F do).

Metrics compared: Hit@5, Recall@5, MRR, Faithfulness, Answer Relevance,
Citation Accuracy.

Writes results to evaluation/results.csv. NEVER fabricate numbers here —
every row must come from an actual run of that variant against a live
LM Studio model and a populated index. If a variant cannot be run yet
(e.g. dependencies not implemented), skip it and say so, rather than
inventing a placeholder score.

Run with:
    python evaluation/ablation.py

Output: evaluation/results.csv, plus a printed summary table
"""

import sys
import os
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_CSV_PATH = "evaluation/results.csv"

ABLATION_VARIANTS = [
    "A_dense_only",
    "B_bm25_only",
    "C_hybrid",
    "D_hybrid_plus_reranker",
    "E_hybrid_plus_reranker_plus_evidence_gate",
    "F_full_pipeline",
]


def build_retrieval_fn_for_variant(variant: str):
    """
    Return a retrieval function (question_text -> list of chunk metadata)
    configured for the given ablation variant.

    D/E/F all retrieve identically at the retrieval-metrics level (hybrid
    search -> cross-encoder rerank -> top 5); what differs between them is
    downstream of retrieval (evidence gate, query reformulation, claim
    verification), which only affects generation/grounding metrics, not
    Hit@5/Recall@5/MRR. Those downstream stages are evaluated separately,
    only for F, using the real end-to-end pipeline (src.pipeline.answer_question).
    """
    from src.dense_retrieval import dense_search
    from src.bm25 import BM25Index
    from src.hybrid_retrieval import hybrid_search
    from src.reranker import rerank
    from src.config import BM25_INDEX_PATH

    if variant == "A_dense_only":
        return lambda q: [c.metadata for c in dense_search(q)]

    if variant == "B_bm25_only":
        bm25_index = BM25Index.load(BM25_INDEX_PATH)
        return lambda q: [c.metadata for c in bm25_index.search(q)]

    if variant == "C_hybrid":
        bm25_index = BM25Index.load(BM25_INDEX_PATH)
        return lambda q: [c.metadata for c in hybrid_search(q, bm25_index)]

    if variant in ("D_hybrid_plus_reranker", "E_hybrid_plus_reranker_plus_evidence_gate", "F_full_pipeline"):
        bm25_index = BM25Index.load(BM25_INDEX_PATH)

        def _retrieve(q: str) -> list[dict]:
            hybrid_candidates = hybrid_search(q, bm25_index)
            reranked = rerank(q, hybrid_candidates)
            return [c.metadata for c in reranked]

        return _retrieve

    raise ValueError(f"Unknown ablation variant: {variant}")


def run_ablation() -> list[dict]:
    """
    Run every variant in ABLATION_VARIANTS through evaluate_retrieval /
    evaluate_generation / evaluate_grounding as applicable, and collect
    one result row per variant.

    Generation/grounding metrics (Faithfulness, Answer Relevance,
    Citation Accuracy) only apply to F_full_pipeline, since A-E stop at
    retrieval/reranking in this project's architecture and never produce
    a generated answer of their own to score. Rows for A-E leave those
    columns blank rather than inventing a number.
    """
    from evaluation.evaluate_retrieval import evaluate_retrieval, load_questions as load_retrieval_questions
    from evaluation.evaluate_generation import evaluate_generation
    from evaluation.evaluate_grounding import evaluate_grounding

    rows: list[dict] = []

    for variant in ABLATION_VARIANTS:
        row: dict = {
            "variant": variant,
            "hit_at_5": None,
            "recall_at_5": None,
            "mrr": None,
            "faithfulness": None,
            "answer_relevance": None,
            "citation_accuracy": None,
        }

        try:
            retrieval_fn = build_retrieval_fn_for_variant(variant)
            retrieval_metrics = evaluate_retrieval(retrieval_fn, load_retrieval_questions())
            row["hit_at_5"] = retrieval_metrics["hit_at_5"]
            row["recall_at_5"] = retrieval_metrics["recall_at_5"]
            row["mrr"] = retrieval_metrics["mrr"]
        except Exception as exc:  # noqa: BLE001 - report and continue with other variants
            print(f"[ablation] Skipping retrieval metrics for {variant}: {exc}")

        if variant == "F_full_pipeline":
            try:
                from src.pipeline import answer_question

                gen_metrics = evaluate_generation(answer_question)
                row["faithfulness"] = gen_metrics["faithfulness"]
                row["answer_relevance"] = gen_metrics["answer_relevance"]

                grounding_metrics = evaluate_grounding(answer_question)
                row["citation_accuracy"] = grounding_metrics["citation_accuracy"]
            except Exception as exc:  # noqa: BLE001
                print(f"[ablation] Skipping generation/grounding metrics for {variant}: {exc}")

        rows.append(row)

    return rows


def write_results_csv(rows: list[dict], path: str = RESULTS_CSV_PATH) -> None:
    """
    Write ablation results to CSV with columns:
        variant, hit_at_5, recall_at_5, mrr, faithfulness,
        answer_relevance, citation_accuracy
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["variant", "hit_at_5", "recall_at_5", "mrr", "faithfulness", "answer_relevance", "citation_accuracy"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    rows = run_ablation()
    write_results_csv(rows)
    for row in rows:
        print(row)

"""
evaluation/evaluate_generation.py

Measures generation quality against evaluation/questions.json:
    Faithfulness         - does the generated answer stay grounded in the
                            retrieved evidence (reuses/extends
                            src/verification.py's claim-checking logic)
    Answer Relevance     - does the generated answer actually address the
                            question asked
    Context Precision/
    Context Recall        - of the retrieved chunks, how many were
                            actually relevant / how much of the needed
                            evidence was retrieved (where applicable —
                            some questions may not have a clean ground
                            truth for this)

If an evaluation framework such as Ragas is used for any of these
metrics, keep its usage modular (isolated to this file) and document what
each metric measures in plain language, per project step 25. Do not
fabricate scores — every reported number must come from an actual run
against a live LM Studio model and a populated index.

Run with:
    python evaluation/evaluate_generation.py

Input:  evaluation/questions.json
Output: printed metrics
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_questions(path: str = "evaluation/questions.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def score_faithfulness(answer: str, evidence_texts: list[str]) -> float:
    """
    Score how well the answer's claims are supported by the evidence.

    Reuses src/verification.py's claim extraction + word-overlap checking
    and reports supported_count / total_count as the faithfulness score.
    Wraps evidence_texts into throwaway RerankedCandidate objects so this
    function can be called with plain text, independent of a live
    pipeline run.
    """
    from src.verification import extract_claims, verify_claim
    from src.reranker import RerankedCandidate

    evidence = [
        RerankedCandidate(chunk_id=f"eval_{i}", relevance_score=1.0, text=t, metadata={})
        for i, t in enumerate(evidence_texts)
    ]

    claims = extract_claims(answer)
    if not claims:
        return 0.0

    checks = [verify_claim(c, evidence) for c in claims]
    supported = sum(1 for c in checks if c.status.value == "SUPPORTED")
    return supported / len(claims)


def score_answer_relevance(question: str, answer: str) -> float:
    """
    Score how directly the answer addresses the question asked, using
    cosine similarity between the question and answer embeddings (via
    src/embeddings.py's bge model). This is a proxy for relevance, not a
    guarantee of correctness — a fluent but wrong answer can still score
    reasonably well.
    """
    import math

    from src.embeddings import embed_query

    q_vec = embed_query(question)
    a_vec = embed_query(answer)

    dot = sum(a * b for a, b in zip(q_vec, a_vec))
    norm_q = math.sqrt(sum(a * a for a in q_vec))
    norm_a = math.sqrt(sum(a * a for a in a_vec))

    if norm_q == 0 or norm_a == 0:
        return 0.0

    return dot / (norm_q * norm_a)


def evaluate_generation(pipeline_fn, questions: list[dict] | None = None) -> dict:
    """
    Run pipeline_fn(question_text) -> PipelineResult for every answerable
    question and aggregate faithfulness / relevance scores.
    """
    if questions is None:
        questions = [q for q in load_questions() if q.get("is_answerable", True)]

    faithfulness_scores = []
    relevance_scores = []

    for q in questions:
        result = pipeline_fn(q["question"])
        if getattr(result, "refused", False):
            continue

        # PipelineResult.verification_result already holds per-claim support
        # status computed against the real evidence used for this answer, so
        # faithfulness is derived from it directly rather than recomputing.
        vr = result.verification_result
        if vr is not None and vr.total_count > 0:
            faithfulness_scores.append(vr.supported_count / vr.total_count)

        relevance_scores.append(score_answer_relevance(q["question"], result.answer))

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "faithfulness": _avg(faithfulness_scores),
        "answer_relevance": _avg(relevance_scores),
        "num_questions": len(questions),
        "num_scored": len(relevance_scores),
    }


if __name__ == "__main__":
    from src.pipeline import answer_question

    results = evaluate_generation(answer_question)
    print(results)

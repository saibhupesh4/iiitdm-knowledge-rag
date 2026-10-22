"""
evaluation/evaluate_grounding.py

Measures grounding quality specifically (as distinct from generic
generation quality in evaluate_generation.py):

    Citation Accuracy      - fraction of citations shown to the user that
                              correspond to a real, retrieved chunk
                              (source + page actually exists in the
                              evidence used for that answer)
    Supported Claim Ratio  - supported claims / total factual claims,
                              aggregated across all evaluated questions
                              (per-question version lives in
                              src/verification.py)
    Correct Refusal Rate   - of the questions marked is_answerable=false
                              in questions.json, what fraction did the
                              system correctly refuse rather than
                              hallucinate an answer for

Run with:
    python evaluation/evaluate_grounding.py

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


def citation_accuracy(citations: list, evidence_used) -> float:
    """
    Check that every citation shown to the user maps back to a chunk that
    was genuinely part of the evidence used for the answer.

    `evidence_used` is the list of RerankedCandidate (or similar objects
    with .metadata) that were actually retrieved for the answer. A
    citation is "accurate" if its (source_filename, page) pair exists
    among that evidence.
    """
    if not citations:
        return 0.0

    evidence_keys = {
        (c.metadata.get("source"), c.metadata.get("page")) for c in evidence_used
    }

    accurate = sum(
        1 for c in citations if (c.source_filename, c.page) in evidence_keys
    )
    return accurate / len(citations)


def correct_refusal_rate(pipeline_fn, unanswerable_questions: list[dict]) -> float:
    """
    Fraction of unanswerable questions (is_answerable=false) for which the
    pipeline correctly refused (PipelineResult.refused == True) instead of
    generating an answer.
    """
    if not unanswerable_questions:
        return 0.0

    correct = 0
    for q in unanswerable_questions:
        result = pipeline_fn(q["question"])
        if getattr(result, "refused", False):
            correct += 1
    return correct / len(unanswerable_questions)


def evaluate_grounding(pipeline_fn, questions: list[dict] | None = None) -> dict:
    """
    Top-level entry point aggregating citation accuracy, supported claim
    ratio, and correct refusal rate across evaluation/questions.json.
    """
    if questions is None:
        questions = load_questions()

    answerable = [q for q in questions if q.get("is_answerable", True)]
    unanswerable = [q for q in questions if not q.get("is_answerable", True)]

    supported_total = 0
    claim_total = 0
    citation_scores = []

    for q in answerable:
        result = pipeline_fn(q["question"])
        if getattr(result, "refused", False):
            continue

        vr = result.verification_result
        if vr is not None:
            supported_total += vr.supported_count
            claim_total += vr.total_count

        if result.citations:
            citation_scores.append(
                citation_accuracy(result.citations, result.evidence)
            )

    supported_claim_ratio = supported_total / claim_total if claim_total else 0.0
    avg_citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0

    return {
        "citation_accuracy": avg_citation_accuracy,
        "supported_claim_ratio": supported_claim_ratio,
        "correct_refusal_rate": correct_refusal_rate(pipeline_fn, unanswerable),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
    }


if __name__ == "__main__":
    from src.pipeline import answer_question

    results = evaluate_grounding(answer_question)
    print(results)

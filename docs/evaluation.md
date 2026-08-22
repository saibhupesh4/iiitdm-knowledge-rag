# Evaluation Plan

This document describes how retrieval, generation, and grounding quality
will be measured. No scores are reported here — this is the plan, not
results. Results only exist once `evaluation/*.py` scripts have actually
been run against a live LM Studio model and a populated index, and are
written to `evaluation/results.csv`.

## Dataset

`evaluation/questions.json` — target 40-50 manually designed questions
covering:

- Academic (curriculum, regulations, calendar)
- Hostel (rules, leave procedure)
- Placement (eligibility, process)
- Institute (general information)
- Unanswerable questions (should be correctly refused)
- Edge cases: exact numbers, dates, course codes, names, terminology

The file currently contains a small set of **example/placeholder**
questions demonstrating the schema — see the `_comment` field in that
file. Real questions with verified `expected_source` / `expected_page`
values still need to be written against the actual ingested documents.

## Retrieval metrics (`evaluation/evaluate_retrieval.py`)

| Metric   | What it measures |
|----------|-------------------|
| Hit@5    | Did the expected source/page appear anywhere in the top-5 retrieved chunks? |
| Recall@5 | Same signal as Hit@5 given one relevant chunk per question in this dataset; tracked separately for clarity and in case multi-chunk ground truth is added later |
| MRR      | Mean reciprocal rank of the first correct chunk across all questions |

## Generation metrics (`evaluation/evaluate_generation.py`)

| Metric           | What it measures |
|------------------|-------------------|
| Faithfulness     | Whether the answer's claims are supported by retrieved evidence |
| Answer Relevance | Whether the answer actually addresses the question asked |
| Context Precision/Recall | Where applicable: how much of the retrieved context was actually relevant / how much needed evidence was retrieved |

If a framework such as Ragas is used for any of these, its usage is kept
modular (isolated to `evaluate_generation.py`) and each metric it reports
is explained in plain language alongside the number.

## Grounding metrics (`evaluation/evaluate_grounding.py`)

| Metric                 | What it measures |
|-------------------------|-------------------|
| Citation Accuracy       | Fraction of shown citations that map to an actually-retrieved chunk |
| Supported Claim Ratio   | supported claims / total factual claims, aggregated across questions |
| Correct Refusal Rate    | Of the unanswerable questions, how many were correctly refused instead of answered |

## Ablation study (`evaluation/ablation.py`)

Compares six pipeline configurations on the same question set:

1. Dense Retrieval only
2. BM25 only
3. Hybrid Retrieval (dense + BM25 fusion)
4. Hybrid + Cross-Encoder Reranking
5. Hybrid + Reranking + Evidence Sufficiency Gate
6. Full Pipeline (adds query reformulation + claim verification)

Compared across: Hit@5, Recall@5, MRR, Faithfulness, Answer Relevance,
Citation Accuracy. Output written to `evaluation/results.csv`, with a
comparison table/plot generated from that CSV — never from invented
numbers.

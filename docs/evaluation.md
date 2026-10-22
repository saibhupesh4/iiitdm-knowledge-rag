# Evaluation Plan

This is a small smoke set, not a benchmark. No scores are reported here.
The current questions use source filenames and pages checked against the
repository PDFs. A run still needs the local embedding/reranker models and
LM Studio.

## Dataset

`evaluation/questions.json` currently covers hostel leave, weekend leave,
placement registration and eligibility, and a question outside the
collection. It is intentionally small. Expected source/page values are
for retrieval checks; the absent question is used for refusal checks.

## Retrieval metrics (`evaluation/evaluate_retrieval.py`)

| Metric   | What it measures |
|----------|-------------------|
| Hit@5    | Did the expected source/page appear anywhere in the top-5 retrieved chunks? |
| Recall@5 | Same signal as Hit@5 given one relevant chunk per question in this dataset; tracked separately for clarity and in case multi-chunk ground truth is added later |
| MRR      | Mean reciprocal rank of the first correct chunk across all questions |

## Generation metrics (`evaluation/evaluate_generation.py`)

| Metric           | What it measures |
|------------------|-------------------|
| Faithfulness proxy | Lexical claim overlap check from `src/verification.py`; it can miss paraphrases and does not prove truth. |
| Answer Relevance proxy | Cosine similarity between question and answer embeddings; it is not a correctness measure. |

These scripts report proxies, not a substitute for manually reviewing
answers and retrieved chunks. No Ragas framework is used.

## Grounding metrics (`evaluation/evaluate_grounding.py`)

| Metric                 | What it measures |
|-------------------------|-------------------|
| Citation Accuracy       | Fraction of shown source/page pairs found in the evidence returned with that answer |
| Supported Claim Ratio   | supported claims / total factual claims, aggregated across questions |
| Correct Refusal Rate    | Of the unanswerable questions, how many were correctly refused instead of answered |

## Ablation study (`evaluation/ablation.py`)

The script compares retrieval configurations and runs end-to-end generation
metrics only for its full-pipeline row. It does not independently toggle
every downstream stage, so its name should not be read as a complete causal
ablation.

It currently lists these retrieval configurations:

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

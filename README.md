# IIITDM Knowledge RAG

**Status: implemented.** Every module under `src/`, `scripts/`, and
`evaluation/` has a working implementation, wired end-to-end through
`src/pipeline.py`. Nothing has been run against real IIITDM PDFs or a
live LM Studio server yet — that's the next step (add PDFs to
`data/raw/<category>/`, run `scripts/rebuild_index.py`, start LM Studio,
run `scripts/test_lm_studio.py`, then `streamlit run app.py`).

## Overview

A document-grounded question-answering system over **official IIITDM
Kancheepuram documents** — academic regulations, curriculum, academic
calendar, hostel rules, hostel leave rules, placement information, and
general institute information.

This is a real hybrid-retrieval RAG pipeline with reranking, an evidence
sufficiency gate, bounded query reformulation, and post-generation claim
verification — not a simple "chat with PDF" wrapper.

## Problem statement

Students need quick, trustworthy answers to questions covered by
official IIITDM documents, with answers traceable back to a specific
document and page — not generic or hallucinated responses.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline
diagram and rationale. Summary:

```
PDFs -> ingest -> clean -> chunk -> embed
                                      |
                                      v
                          Chroma (dense) + BM25 (lexical)
                                      |
        query -> category detection -+-> dense search
                                      +-> BM25 search
                                      |
                          normalize + fuse -> top 20
                                      |
                        cross-encoder rerank -> top 5
                                      |
                       evidence sufficiency gate
                     (insufficient -> reformulate -> retry once)
                                      |
                          grounded prompt -> LM Studio (local LLM)
                                      |
                            claim/evidence verification
                                      |
                       final answer + citations -> Streamlit
```

## Main features

- Hybrid dense + BM25 retrieval with explicit score fusion
- Cross-encoder reranking (top-20 -> top-5)
- Evidence sufficiency gate with bounded query reformulation
- Fully local generation via LM Studio (no external API keys)
- Post-generation claim/evidence verification
- Page-level citations, never fabricated
- Streamlit UI with retrieval and grounding transparency panels
- Offline evaluation + a 6-way ablation study

## Folder structure

```
iiitdm-knowledge-rag/
├── data/                  raw PDFs by category + processed cache
├── storage/               persisted Chroma + BM25 indexes
├── src/                   all RAG pipeline modules
├── scripts/                ingestion / indexing / LM Studio test CLIs
├── evaluation/             questions, metric scripts, ablation, results
├── docs/                   architecture, setup, evaluation, LM Studio docs
├── app.py                  Streamlit UI
└── requirements.txt
```

See [`docs/architecture.md`](docs/architecture.md) for what each `src/`
module is responsible for.

## Technology stack

Python · LangChain · Chroma · BM25 (rank_bm25) · Sentence Transformers
(BGE-small embeddings) · Cross-Encoder reranker · LM Studio (local LLM) ·
Streamlit

No LangGraph, no multi-agent framework, no GraphRAG, no cloud
infrastructure — see `docs/architecture.md` for the explicit
out-of-scope list.

## Local setup

See [`docs/setup.md`](docs/setup.md) for full instructions. Quick
version:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit LM_STUDIO_MODEL
python scripts/test_lm_studio.py
```

## LM Studio setup

See [`docs/lm_studio.md`](docs/lm_studio.md). No OpenAI/Gemini/Anthropic
API key is used anywhere — generation runs entirely through LM Studio's
local OpenAI-compatible server.

## Where to put PDFs

`data/raw/academic/`, `data/raw/hostel/`, `data/raw/placement/`,
`data/raw/institute/` — see [`data/README.md`](data/README.md) for
expected metadata fields.

## How indexing works

`python scripts/rebuild_index.py` runs ingestion -> cleaning -> chunking,
then builds and persists both the Chroma vector index and the BM25 index
under `storage/`. The app loads these persisted indexes rather than
rebuilding them on every startup.

## How retrieval works

Every query runs through category detection, dense search, BM25 search,
score normalization, and fusion to produce a top-20 hybrid candidate
list, which a cross-encoder reranks down to a top-5. See
`docs/architecture.md` for the full flow including the evidence
sufficiency gate and bounded query reformulation.

## How Streamlit works

`app.py` calls `src/pipeline.answer_question()` and renders the result:
answer, sources, and four expanders (Retrieval Details, Retrieved
Sources, Grounding Check, Technical Details). No RAG logic lives in
`app.py` itself.

## Evaluation plan

See [`docs/evaluation.md`](docs/evaluation.md). Retrieval metrics
(Hit@5, Recall@5, MRR), generation metrics (Faithfulness, Answer
Relevance), and grounding metrics (Citation Accuracy, Supported Claim
Ratio, Correct Refusal Rate) are computed by the scripts under
`evaluation/` — none reported until actually measured against a live
LM Studio model and a populated index.

## Ablation study

Six pipeline variants (dense only -> BM25 only -> hybrid -> +reranker ->
+evidence gate -> full pipeline) compared on the same question set,
written to `evaluation/results.csv` by `evaluation/ablation.py`. Only
`F_full_pipeline` reports generation/grounding metrics, since variants
A-E stop at retrieval/reranking in this architecture and never produce a
generated answer of their own to score.

## Limitations

- Every module is implemented, but nothing has been run yet against real
  IIITDM documents or a live LM Studio server — `evaluation/questions.json`
  still contains placeholder example questions (see its `_comment` field)
  that must be replaced with 40-50 real, manually-verified questions
  before evaluation numbers mean anything.
- Claim verification uses lexical word-overlap between each claim and
  the evidence, which is explainable and fast but is a proxy for
  semantic support, not a guarantee of correctness. It reduces, but does
  not eliminate, unsupported claims — stated explicitly rather than
  claimed as a solved hallucination problem.
- Retrieval quality depends entirely on which official documents are
  ingested; questions about anything not in `data/raw/` should be
  refused, not guessed at.
- Local model quality (embedding, reranker, and LLM) is bounded by what
  runs reasonably on the available hardware — this is a deliberate
  tradeoff for a fully local, no-API-key system.

## LangSmith Observability

LangSmith is an **optional** tracing/observability layer sitting *outside*
the core RAG decision flow (see `docs/architecture.md`). It is never
required — the app works exactly the same, with or without it, when
tracing is disabled or unavailable.

To enable it:

1. Set `LANGSMITH_TRACING=true` in your `.env` (copy from `.env.example`).
2. Add your `LANGSMITH_API_KEY` (get one from [smith.langchain.com](https://smith.langchain.com)).
3. Set `LANGSMITH_PROJECT` (defaults to `iiitdm-knowledge-rag`).
4. Install dependencies (`langsmith` is now in `requirements.txt`) and run
   the app as usual.

What gets traced when enabled:

- `answer_question()` in `src/pipeline.py` — the root trace for a question
- `detect_category()` — query understanding
- `hybrid_search()` — dense + BM25 retrieval and fusion
- `rerank()` — cross-encoder reranking
- `check_sufficiency()` — the evidence sufficiency gate's decision
- `reformulate_query()` — query rewrite on a weak-evidence retry
- The LM Studio chat completion call itself (via `langsmith.wrappers.wrap_openai`
  around the OpenAI-SDK client in `src/llm.py`)
- `verify_answer()` — claim/evidence verification

LM Studio remains the actual (local) LLM in every case — LangSmith only
observes the calls, it never replaces LM Studio or requires an OpenAI
API key. If tracing is off, misconfigured, or LangSmith is unreachable,
the RAG pipeline still runs normally; observability failures are not
allowed to break question answering.

## Future improvements

- Expand the document set beyond the initial 10-20 files
- Tune fusion weights and evidence-gate thresholds against the ablation
  results once real data exists
- Consider category-aware metadata filtering in retrieval, not just
  category detection
- Revisit the reranker/embedding model choice if local hardware allows
  a stronger option

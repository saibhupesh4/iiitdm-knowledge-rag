# IIITDM Knowledge RAG

A local Streamlit app for answering questions from IIITDM Kancheepuram PDF documents. It retrieves evidence from a Chroma vector index and a BM25 keyword index, then sends the selected passages to a local model served by LM Studio.

## Features

- Hybrid retrieval: semantic search plus keyword search, combined with reciprocal rank fusion.
- Local cross-encoder reranking and an evidence threshold before generation.
- Page metadata carried into displayed citations.
- Refusal when evidence is insufficient or the generated answer fails the project’s lexical support check.
- Separate routes: ordinary questions go to the local model; IIITDM and institute-document questions use RAG and citations.
- Streamlit interface with retrieval and grounding details.

## Architecture

`app.py` calls `src.pipeline.answer_question()`. The pipeline retrieves and reranks evidence, builds a grounded prompt, calls the LM Studio local endpoint, checks answer claims, and returns the answer and source pages. Indexing is a separate operation. See [`docs/architecture.md`](docs/architecture.md) and [`docs/flow.md`](docs/flow.md).

## Requirements

- Python 3.10 or newer (the source uses Python 3.10 type syntax; use a Python version supported by the installed PyTorch and Streamlit packages).
- LM Studio with a loaded chat/instruct model and its local server enabled.
- Official PDFs in `data/raw/<category>/` for the desired categories.

## Install

From the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Model weights for embeddings and reranking may need to be downloaded the first time those components run.

## Configure LM Studio

Start LM Studio’s local server (default URL `http://127.0.0.1:1234/v1`) and set `LM_STUDIO_MODEL` in `.env` to the exact model ID shown by that server. The sample `.env.example` uses a placeholder; replace it before asking questions. To check the server:

```powershell
python scripts/test_lm_studio.py
```

The OpenAI Python package is used as an HTTP client for this local endpoint. Generation does not use a hosted model API.

## Build the indexes

The repository already contains persisted indexes. Rebuild only after changing the PDFs or indexing configuration:

```powershell
python scripts/rebuild_index.py
```

This reloads PDFs, cleans and chunks their text, recreates the Chroma collection, and writes `storage/bm25_index.pkl`. Rebuilding replaces the current indexes; keep a backup if you need to preserve them. The app does not rebuild indexes on startup.

## Run

```powershell
streamlit run app.py
```

## Project layout

- `app.py` — Streamlit interface.
- `src/pipeline.py` — question-answering orchestration.
- `src/` — ingestion, retrieval, prompt, LM Studio, citation, and verification modules.
- `scripts/rebuild_index.py` — document indexing entry point.
- `scripts/ingest.py` — inspect ingestion and chunking without changing indexes.
- `data/raw/` — source PDFs by category.
- `storage/` — persisted Chroma and BM25 indexes.
- `docs/` — architecture, flow, frontend, design decisions, setup, LM Studio, and evaluation notes.

## Limitations

Retrieval and generation quality depend on PDF extraction, the indexed documents, locally available model weights, and LM Studio’s loaded model. The evidence gate and answer check use model scores and lexical overlap; neither proves that a statement is true. The local Python environment used for this cleanup does not have the project dependencies installed, so live retrieval and LM Studio answer quality were not measured here. Evaluation questions are examples, not verified ground truth.

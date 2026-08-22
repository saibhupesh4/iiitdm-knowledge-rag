# Architecture

This document describes the locked architecture for IIITDM Knowledge RAG.
It is a document-grounded question-answering system over official IIITDM
Kancheepuram documents — not a generic "chat with PDF" tool.

## Full pipeline

```
OFFICIAL IIITDM PDFs
        |
        v
PDF INGESTION                          (src/ingestion.py)
        |
        v
TEXT CLEANING + METADATA               (src/cleaning.py)
        |
        v
STRUCTURE-AWARE RECURSIVE CHUNKING     (src/chunking.py)
        |
        v
BGE-SMALL EMBEDDINGS                   (src/embeddings.py)
        |
        v
CHROMA + BM25 INDEX                    (src/dense_retrieval.py, src/bm25.py)
        |
        v
USER QUERY
        |
        v
QUERY UNDERSTANDING                    (src/query_understanding.py)
        |
        v
   DENSE SEARCH  +  BM25 SEARCH        (src/dense_retrieval.py, src/bm25.py)
        |
        v
SCORE NORMALIZATION                    (src/hybrid_retrieval.py)
        |
        v
SCORE FUSION                           (src/hybrid_retrieval.py)
        |
        v
HYBRID TOP-20                          (src/hybrid_retrieval.py)
        |
        v
CROSS-ENCODER RERANKER                 (src/reranker.py)
        |
        v
TOP-5
        |
        v
EVIDENCE SUFFICIENCY                   (src/evidence_gate.py)
        |
        v
QUERY REFORMULATION IF WEAK            (src/query_reformulation.py)
   (bounded retry, MAX_QUERY_REFORMULATION_RETRIES = 1)
        |
        v
GROUNDED PROMPT                        (src/prompt.py)
        |
        v
LM STUDIO (LOCAL LLM)                  (src/llm.py)
        |
        v
DRAFT ANSWER
        |
        v
CLAIM / EVIDENCE CHECK                 (src/verification.py)
        |
        v
FINAL ANSWER / REFUSAL
        |
        v
CITATIONS                              (src/citations.py)
        |
        v
STREAMLIT                              (app.py)
        |
        v
EVALUATION                             (evaluation/evaluate_*.py)
        |
        v
ABLATION                               (evaluation/ablation.py)
```

`src/pipeline.py` is the single orchestration point that wires every
module above together for a live query. `app.py` calls only
`pipeline.answer_question()` — it does not call the individual retrieval,
generation, or verification modules directly.

## LangSmith: a side observability layer, not a pipeline component

LangSmith sits *beside* the pipeline above, not inside its decision
flow:

```
            IIITDM RAG PIPELINE (unchanged, see diagram above)
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
    RETRIEVAL          RERANKING         GENERATION
        |                 |                 |
        +-----------------+-----------------+
                          |
                          v
                     LANGSMITH
                OBSERVABILITY LAYER  (optional)
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
       TRACES          LATENCY        DEBUGGING /
                                       EVALUATION
```

Concretely, `@traceable` decorates the major logical steps
(`answer_question`, `detect_category`, `hybrid_search`, `rerank`,
`check_sufficiency`, `reformulate_query`, `verify_answer`), and the LM
Studio OpenAI-SDK client is wrapped with `langsmith.wrappers.wrap_openai`
in `src/llm.py` so LM Studio calls show up as generation spans too.

None of this changes what any function returns or how a decision is
made — `@traceable` only observes. The pipeline must, and does, keep
working with `LANGSMITH_TRACING=false` or LangSmith otherwise
unavailable.

## Why hybrid retrieval

Dense (embedding-based) retrieval is strong at semantic similarity but
can miss exact terms that matter in official documents: course codes,
policy names, percentages, dates, form names, abbreviations. BM25
(lexical) retrieval is strong at exactly these exact-match cases. Fusing
both, rather than picking one, is intended to cover both failure modes.

## Why an evidence sufficiency gate

Not every question can be answered from the available documents. Rather
than always generating something, the pipeline checks whether the
reranked evidence actually clears a relevance threshold before
generation is attempted at all. If it doesn't, one bounded query
reformulation + retry is attempted; if that still doesn't produce
sufficient evidence, the system refuses rather than guesses.

## Why claim verification after generation

Even with grounded evidence, an LLM can still produce a claim not
actually supported by what was retrieved. Verification re-checks the
draft answer's claims against the evidence before showing it to the
user. This is described as an attempt to reduce unsupported claims — not
as a guarantee that hallucination is eliminated.

## Explicitly out of scope

Per the locked architecture, this project does NOT use: LangGraph,
multi-agent architectures, GraphRAG, multimodal RAG, Kubernetes, cloud
infrastructure, unnecessary databases, or unnecessary external APIs.

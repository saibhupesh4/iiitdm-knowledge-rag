# Failure Analysis and Database Guide

This document explains where the current IIITDM Knowledge RAG implementation can fail, how to find the failing stage, and what ChromaDB and the other persisted data stores do in this repository. It describes the code as it exists; a suggested recovery is not a claim that the application performs that recovery automatically.

For the end-to-end architecture and workflow, see [architecture](architecture.md) and [flow](flow.md). For a longer learning guide, see [`to_make.md`](../to_make.md). This repository does not contain a `docs/know.md` file.

## 1. What “the database” means in this project

There is no single database that handles everything. The project uses two different persisted retrieval stores, plus the original PDFs. The model weights and the Python packages are separate from those stores.

| Component | What it stores or computes | Why it exists | Created by | Used by |
|---|---|---|---|---|
| Original PDFs | Official source documents under `data/raw/academic/`, `hostel/`, `placement/`, and `institute/` | Human-readable source of truth from which the indexes can be rebuilt | User/project data | `src.ingestion.load_all_documents()` and `scripts/rebuild_index.py` |
| ChromaDB collection `iiitdm_docs` | Chunk text, IDs, metadata, and vector embeddings in a local persistent Chroma collection | Semantic/vector retrieval: find passages with similar meaning even when wording differs | `src.dense_retrieval.index_chunks()` after `src.embeddings.get_embedding_model()` embeds documents | `src.dense_retrieval.get_vectorstore()` and `dense_search()` |
| BM25 index | Tokenized corpus and aligned chunk IDs, texts, and metadata, saved as a Python pickle | Lexical retrieval for exact terms, form names, dates, abbreviations, and codes | `BM25Index.build()` and `BM25Index.save()` in `scripts/rebuild_index.py` | `src.pipeline._get_bm25_index()` then `BM25Index.search()` |
| Embedding model | Computes a numeric vector for text; it is not a database | Converts document chunks and queries into compatible vectors | `src.embeddings.get_embedding_model()` | Chroma indexing and `dense_search()` |
| Cross-encoder model | Scores a question and candidate passage together; it is not a database | Reranks the hybrid candidate list | Loaded lazily by `src.reranker.get_reranker_model()` | `src.reranker.rerank()` |
| LM Studio chat model | Generates answer text; it is not a database | Turns the question and selected evidence into a readable response | Loaded and served by LM Studio on the user's computer | `src.llm.generate()` |

The defaults in `src/config.py` are:

- Chroma collection: `iiitdm_docs`
- Chroma persistence directory: `<project root>/storage/chroma`
- BM25 pickle: `<project root>/storage/bm25_index.pkl`
- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

`CHROMA_PATH`, `BM25_INDEX_PATH`, and the model names can be overridden in `.env`. Relative store paths are resolved from the project root, not the shell's current directory. Chroma uses local persistence through `langchain_chroma.Chroma`; the app does not connect to a separately hosted database server.

### ChromaDB’s role and significance

An embedding model converts a text chunk into a vector (a list of numbers that represents the text for similarity search). Chroma stores those vectors with the chunk's text, ID, and metadata, and compares a query vector against stored vectors. In this project that gives the dense retriever a way to find semantically related passages even when the question uses different words from the PDF.

The implementation path is:

1. `scripts/rebuild_index.py` runs `load_all_documents()`, `clean_documents()`, and `chunk_documents()`.
2. `src.dense_retrieval.index_chunks()` calls `get_vectorstore()` and adds chunk documents in batches, with their `chunk_id` values as Chroma IDs. Chroma's embedding function comes from `src.embeddings.get_embedding_model()`.
3. `get_vectorstore()` creates a `Chroma` wrapper with `collection_name=CHROMA_COLLECTION_NAME`, `embedding_function=get_embedding_model()`, and `persist_directory=CHROMA_PATH`. This opens the existing local collection when present, or creates/opens the named collection through the Chroma integration.
4. `dense_search()` calls `similarity_search_with_score(query, k=top_k)` and returns the text, metadata, ID, and a higher-is-better transformed score in each `DenseCandidate`.
5. `src.hybrid_retrieval.hybrid_search()` combines the dense results with BM25 results. Dense search is only one half of retrieval; it does not generate the answer.

The chunk metadata starts in `src.ingestion.load_pdf_pages()`: `source`, `page` (1-based), `category`, and `title`. Cleaning may add `section`; chunking adds `chunk_id`, `chunk_index`, and `chunk_size_chars`. `src.dense_retrieval._sanitize_metadata()` removes `None` values and converts non-primitive values before adding records to Chroma.

**Without Chroma, the current question path loses semantic vector search.** `hybrid_search()` catches a dense-retrieval exception and can continue with BM25, so question answering may still retrieve keyword matches if the BM25 index and reranker work. That is degraded retrieval, not a Chroma replacement; `scripts/rebuild_index.py` still calls Chroma reset and indexing functions and will not be a valid BM25-only rebuild without code changes.

Possible alternatives have different costs:

| Option | What changes | Trade-offs |
|---|---|---|
| Remove Chroma but leave the current design as-is | Dense calls fail and are caught by `hybrid_search`; BM25 may still contribute. The indexing script still expects Chroma. | Less retrieval coverage; confusing degraded mode; rebuild path still needs changes. |
| Replace Chroma with another vector database | Replace the `Chroma` creation, add, reset, and query calls while preserving the `dense_search()` output interface. | Possible different persistence, deployment, filtering, and scaling behavior; more implementation and migration work; existing vectors/collection may need rebuilding. No alternative is automatically better. |
| Keep all vectors in memory | Load or compute vectors and compare them in Python for every query. | Can be simple for a tiny collection, but memory use grows with the corpus; persistence, metadata handling, and efficient search must be implemented or supplied elsewhere. |
| Use only BM25 | Remove dense retrieval from fusion and rely on the existing keyword index. | Simpler and good for exact terms, but weaker when a question paraphrases document language. |

ChromaDB does **not** write the final natural-language answer, understand a question like an LLM, judge whether a passage actually answers the question, prevent hallucinations, or replace prompt construction and the evidence checks. It returns candidates according to vector similarity. Retrieval relevance and answer correctness remain separate questions.

## 2. End-to-end failure map

The stages below follow the two actual workflows. Indexing runs only when `python scripts/rebuild_index.py` is invoked. Query-time failures happen after a user submits a question in `app.py`.

| Failure | Why it happens / what the user may see | Where to inspect | How to diagnose | Possible fix |
|---|---|---|---|---|
| PDF not discovered or fails to load | The PDF is outside the four category folders, has an unexpected extension, is damaged, encrypted, or `PyPDFLoader` raises an error. It may be skipped with an ingestion warning; the relevant text will never be searchable. | `src/ingestion.py`: `CATEGORIES`, `discover_pdfs()`, `load_pdf_pages()`, `load_all_documents()`; `scripts/rebuild_index.py` | Check the actual folder and filename. Run the indexing script and read its “Found … PDF files” and per-file page counts/warnings. Open the PDF and verify it contains selectable text. | Put the PDF in the correct `data/raw/<category>/` folder. Repair/export a readable PDF. If it is scanned, OCR is not currently implemented; use a text-readable source or add OCR as a separate future change. Rebuild only after confirming the source set. |
| Extraction loses or scrambles text | PDF layout, columns, tables, ligatures, or scanned pages can yield broken reading order or little text. The app may refuse, return an incomplete answer, or miss facts. | `src/ingestion.py`: `load_pdf_pages()`; `src/cleaning.py`: `clean_documents()` | Compare extracted `page_content` for the exact source page with what a human sees in the PDF. Do this before changing retrieval. | Use a text-readable copy or improve extraction for the affected format. OCR/table extraction is not part of the current path. Rebuild after extraction changes. |
| Cleaning removes useful content or keeps artifacts | The cleaner removes control characters and standalone page-number lines and normalizes whitespace. Some unusual documents may use a meaningful standalone number; other headers/footers may remain. | `src/cleaning.py`: `remove_pdf_artifacts()`, `normalize_whitespace()`, `is_empty_page()`, `detect_section_heading()` | Inspect the raw loaded page and the cleaned page. Check important dates, numbers, percentages, form names, and rules survived. | Adjust cleaning only for a demonstrated extraction issue. It is deliberately conservative; do not remove broad classes of numbers. Rebuild both indexes after changing cleaning. |
| Important fact is split across chunks | Pages are chunked separately. A fact spanning a page boundary, or a paragraph larger than the chunk limit, may be split. Overlap helps within a page but does not combine pages. | `src/chunking.py`: `build_splitter()`, `chunk_documents()`; `CHUNK_SIZE` and `CHUNK_OVERLAP` in `src/config.py` | Find the fact in the extracted page text, then inspect all chunks from that source/page using their metadata and chunk IDs. | Adjust chunking only if evidence shows a boundary problem. Rebuild the vector and BM25 indexes together; chunk settings must match the stored corpus. |
| Embedding fails while building/querying | The model package/weights are unavailable, a device fails, or model configuration is invalid. Indexing stops or dense retrieval logs an error. | `src/embeddings.py`: `get_embedding_model()`; `src/dense_retrieval.py`: `get_vectorstore()`; `scripts/rebuild_index.py` | Read the model/device error. Confirm the model name and whether its weights are locally available. Test on CPU if GPU initialization is the issue; the embedding module attempts CPU fallback after CUDA failure. | Install dependencies/download the model when possible, free resources, or use the configured CPU fallback. Keep the same embedding model for document and query encoding. A model change requires a controlled rebuild of vectors. |
| Chroma collection is missing, empty, or opened at the wrong path | `get_vectorstore()` is lazy and creates/opens the configured collection. A wrong but writable path may look like a new empty collection rather than a clear “missing DB” error. Dense search then returns no useful candidates. | `src/config.py`: `CHROMA_PATH`, `CHROMA_COLLECTION_NAME`; `src/dense_retrieval.py`: `get_vectorstore()`, `dense_search()`; `.env` | Print/check the resolved `CHROMA_PATH`, confirm the collection name, inspect the expected directory and persisted Chroma files, and compare a known query with the original source. Check whether BM25 still returns candidates; hybrid logs report counts for each retriever. | Correct the `.env` path/collection. Back up the existing storage first. If the index truly is absent or empty, confirm PDFs and model availability, then rebuild the indexes. Do not delete the directory just to test a path. |
| Chroma index and embedding model do not match | Changing `EMBEDDING_MODEL_NAME` without rebuilding can cause a vector dimension error, or can silently compare vectors from incompatible model spaces if dimensions happen to match. | `.env`; `src/config.py`; `src.embeddings.py`; `src.dense_retrieval.get_vectorstore()` and `dense_search()` | Compare the embedding model setting used to build the collection with the current query model. Check Chroma exception/log output for dimensionality errors. | Restore the index's original model setting, or back up the stores and rebuild all vectors using the intended model. Rebuild BM25 in the same run so both indexes represent the same chunks. |
| PDF changed but indexes are stale | The app loads persisted stores; it does not watch PDFs or re-index at startup. Answers can cite an old page or fail to find a new rule. | `scripts/rebuild_index.py`; `src/pipeline._get_bm25_index()`; `src.dense_retrieval.get_vectorstore()` | Compare the current PDF text with retrieved chunk text and dates. Inspect `source`, `page`, and chunk IDs. Confirm when the index was last rebuilt from your own logs/backup rather than assuming it is current. | Back up `storage/`, verify the PDF collection and model settings, then run the rebuild script. It resets the configured Chroma collection and replaces the BM25 pickle. |
| BM25 index missing or unreadable | The app explicitly checks for the configured BM25 file. Pickle incompatibility, corruption, or a wrong path can prevent the RAG pipeline from loading it. The UI shows a runtime error rather than an answer. | `src/pipeline.py`: `_get_bm25_index()`; `src/bm25.py`: `BM25Index.load()`; `BM25_INDEX_PATH` in `src/config.py` | Confirm the resolved path exists. Read the “Failed to load persisted BM25 index” reason. Do not open/unpickle an untrusted file. | Correct the path or restore a known-good backup. If it is corrupt and originals are available, back up the current file and rebuild from PDFs with `scripts/rebuild_index.py`. |
| BM25 returns irrelevant keyword matches | BM25 scores lexical overlap, not whether a passage answers the question. `BM25Index.search()` returns the top scores without a minimum score cutoff; if query terms are weak or absent, the top list can still contain poor candidates. | `src/bm25.py`: `_tokenize()`, `BM25Index.search()`; hybrid candidate logging | Run BM25 search alone and read the candidate text, score, source, and page. Check whether expected source appears at all and whether the query contains useful document terms. | Use the source's terminology in a diagnostic query; confirm extracted text/tokens. If terms exist but rank is poor, review corpus/chunking and evaluate alternatives. Do not assume a nonempty result means relevance. |
| Dense search finds semantically similar but wrong passages | Vector similarity is a nearest-neighbor signal, not a fact checker. Related words/themes can retrieve a passage about a different policy. | `src/dense_retrieval.py`: `dense_search()`; `src.embeddings.py`; returned `DenseCandidate` text/metadata | Inspect the actual returned passage, not just its score. Compare dense-only and BM25-only candidates. Check model consistency. | Fix missing/poor source extraction or tuning only after identifying the retrieval error. Hybrid ranking and reranking may help but cannot recover chunks absent from both top-k lists. |
| Hybrid fusion omits the answer or misranks it | RRF combines only the candidates returned by dense and BM25. A correct chunk outside both top-k lists cannot be fused. Duplicate/missing chunk IDs are ignored or skipped. | `src/hybrid_retrieval.py`: `hybrid_search()`, `fuse_candidates_rrf()`; `DENSE_TOP_K`, `BM25_TOP_K`, `HYBRID_TOP_K` | Inspect each retriever's result and rank, then inspect fused candidates. The `[hybrid]` logs show counts; source/page/text tell whether the expected evidence entered fusion. | Correct the upstream extraction/index problem first. If a valid chunk is being cut off, evaluate a higher candidate limit. Avoid changing RRF or adding components without a measured example. |
| Duplicate filenames cause chunk-ID collisions | `create_chunk_id()` uses source filename, page, and within-page chunk index. If two category folders contain the same filename, corresponding chunks can receive the same ID. Fusion deduplicates by ID; one document's chunk may mask the other. | `src/chunking.py`: `create_chunk_id()`; `src.dense_retrieval.index_chunks()`; `src.hybrid_retrieval.fuse_candidates_rrf()` | Search all `data/raw` category directories for duplicate PDF filenames and inspect repeated chunk IDs. | Rename colliding files so their names are unique across the corpus, then back up and rebuild both indexes. |
| Reranker model cannot load or reranks poorly | Cross-encoder weights may be missing, device initialization may fail, or its training domain may not fit the question. A model load failure raises an error after its device fallback attempts; poor scores/order can drop the useful passage from the final five. | `src/reranker.py`: `get_reranker_model()`, `rerank()`; `RERANKER_MODEL_NAME`, `RERANK_TOP_K` | Inspect raw and converted scores plus source/page/text before and after reranking. Confirm the model can load locally. Compare the expected source's rank before and after reranking. | Make model weights available and use CPU if GPU loading fails. Correct retrieval before changing the reranker. Evaluate any model/limit change on known questions; do not treat a higher score as proof. |
| Evidence gate refuses a supported question or accepts weak evidence | It requires a configured number of unique chunks whose sigmoid-converted reranker score clears a threshold (defaults: 2 and 0.5). The sigmoid score is not a calibrated probability of “answers the question”; two on-topic but non-answering passages may pass. | `src/evidence_gate.py`: `check_sufficiency()`; `src/reranker.py`: `_safe_sigmoid()`; `.env` settings | The UI's Grounding Check shows qualifying count, threshold, and minimum. In debug output inspect the actual qualifying passages, not only the verdict. | First fix retrieval/ranking. Only adjust thresholds after recording false accepts and false refusals from representative documents. Do not claim the gate proves sufficiency. |
| Retry retrieves unrelated passages | If the gate fails, the retry appends fixed terms from `src/query_reformulation.py` based on a rule-based category. Broad extra terms may change ranking and introduce irrelevant chunks. | `src/query_understanding.py`: `detect_category()`; `src/query_reformulation.py`: `CATEGORY_EXPANSIONS`; `PipelineResult.reformulated_query` | Compare original and reformulated queries and their candidate text, using Retrieval Details/debug output. | Improve the original query or correct the source/index. Keep retry count bounded. Change expansion terms only against examples where they improve expected evidence. |
| Useful evidence is missing from the prompt | The prompt builder skips empty chunks and numbers the usable chunks. If upstream text/metadata is absent, the LLM cannot recover it from the original PDF. | `src/prompt.py`: `format_evidence_context()`, `build_prompt()`; `src.pipeline.answer_question()` | Inspect the exact selected candidate text and source labels; call/debug prompt construction during local diagnosis. Ensure the source number in a generated citation refers to the intended chunk. | Fix extraction, retrieval, or metadata. Keep the prompt limited to valid evidence and preserve the numbering relationship. |
| LM Studio is unavailable, model ID is wrong, or output is empty | The local server may be stopped, endpoint/model settings can be wrong, generation can time out or return no visible content. The app displays an error for its handled LM Studio/runtime failures. | `.env`; `src/llm.py`: `_validate_configuration()`, `get_client()`, `generate()`; `scripts/test_lm_studio.py`; `app.py:main()` | Run `python scripts/test_lm_studio.py`. Confirm the local server is started, its URL and loaded model ID match `.env`, and a direct test response has visible text. | Start/load the local model, correct `.env`, and retry. No cloud fallback exists. An empty visible completion is treated as an error; hidden reasoning is not used as the answer. |
| Model adds unsupported facts or misunderstands a condition | The prompt asks for evidence-only answers, but generation is probabilistic. Similar wording can cause a missed exception or unsupported inference. | `src/prompt.py`; `src.llm.py`; `src.verification.py`: `verify_answer()` | Compare each factual sentence with the exact passage and conditions. Check which `[Source N]` it names and what that source actually says. | Narrow the question, fix/expand relevant evidence, or keep the refusal. Prompt instructions and lexical checking reduce risk but cannot guarantee correctness. |
| Verification refuses a correct paraphrase or accepts an unsupported claim | Verification compares sets of meaningful words. One claim passes if at least 50% of its content words overlap a selected chunk; the final `sufficiently_supported` threshold is 80% of claims. Word overlap does not test entailment or negation reliably. | `src/verification.py`: `extract_claims()`, `verify_claim()`, `VerificationResult.sufficiently_supported`; `src/pipeline.answer_question()` | Inspect extracted claim text, `support_score`, cited source indices, and evidence text. Compare semantic meaning manually. | Correct the retrieved passage or answer. Treat this as a rough filter only. Do not present its “supported” label as proof of truth. |
| Citation is missing, points to a confusing page, or repeats a page | Citations are built from the reranked chunks' metadata, one citation per chunk; multiple chunks from one page can therefore yield repeated source/page entries. Bad input metadata remains bad citation metadata. The UI source expander lists citation metadata, not full chunk text. | `src/citations.py`: `build_citations()`; `src/prompt.py`: source numbering; `src.pipeline.PipelineResult`; `app.py`: `render_sources()` and `render_retrieved_sources_expander()` | Match each displayed title/page/filename to the evidence chunk and PDF page. Verify the answer's `[Source N]` against the prompt order. | Fix metadata at ingestion/chunking and rebuild. Read the actual passage; a correct source label alone does not prove the passage supports the answer. |
| UI shows an error or no useful diagnostic | Streamlit catches `LMStudioUnavailableError` and `RuntimeError` in `main()`. The UI renders answer/source/diagnostic fields but does not run retrieval itself; some exceptions outside those handled types may still surface as an app error. The “Retrieved Sources” expander currently displays citation metadata rather than passage text. | `app.py`: `main()`, `render_answer()`, `render_sources()`, expanders; `src.pipeline.answer_question()` | Read the exact displayed error and the terminal where Streamlit is running. Then inspect the corresponding backend stage before changing UI code. | Fix the source stage identified by the error. Do not assume the interface is responsible for an upstream retrieval or model failure. |

## 3. Why answers can look random or irrelevant

“Random answer” is an observation, not a diagnosis. Several different faults can produce a fluent but wrong response. Use the first point where the expected information disappears to identify the class of failure.

| Failure type | Example | How to distinguish it |
|---|---|---|
| Ingestion failure | The PDF page says a leave form is due 48 hours ahead, but extracted text is empty or garbled. | The fact is missing from extracted/cleaned text and every index. Start at the PDF and `load_pdf_pages()`. |
| Retrieval failure | The fact exists in chunks, but neither dense nor BM25 top results contain it. | Search each retriever independently. The fact is in the corpus but not in returned candidates. |
| Ranking failure | BM25 or dense search returns the right page, but fusion/reranking drops it below the final top five. | Compare ranks at each stage: dense, BM25, fused, reranked. The chunk appears earlier and disappears later. |
| Evidence failure | The selected chunks are on the hostel topic but do not contain the requested exception or answer. The gate still accepts two chunks over 0.5. | Read all final passages and check whether they directly answer the asked point. A “sufficient” gate verdict does not prove this. |
| Prompt failure | The right chunks were retrieved, but the prompt omits them, truncates/renumbers them unexpectedly, or the cited source label is mismatched. | Inspect `format_evidence_context()` output and compare source labels with candidate order. |
| Generation failure | The correct page is present in the prompt, but the answer changes a condition or adds a rule absent from it. | The evidence is correct in the prompt; the unsupported claim first appears in generated text. Review the claim manually and inspect the verifier's lexical score. |
| Citation failure | The answer text may match a passage, but the displayed file/page is wrong, repeated, or absent. | Compare the citation's source/page metadata to the exact chunk and the original PDF page. |
| Configuration failure | `.env` points to an empty Chroma directory or query embeddings use a different model from indexing. | Inspect resolved paths, collection name, and model settings; compare them with the settings used for the stored index. |

An LLM cannot reliably supply evidence that retrieval did not pass to it. On the RAG route, the prompt explicitly says to answer only from provided evidence. If the evidence is missing, a cooperative model should refuse; a model that answers anyway can hallucinate. A larger model may follow instructions better in some cases, but it cannot make a wrong or missing page become correct evidence. More components also mean more stages where content can be lost, misranked, or misinterpreted. Add or change a component only after identifying the failing stage with an example.

### Diagnostic order for an irrelevant answer

1. Open the cited/original PDF page and verify that it actually contains the requested fact.
2. Check the page text extracted by `src.ingestion.py`; compare it with the PDF.
3. Check the cleaned page and the chunks for that filename/page.
4. Search dense and BM25 separately. See whether the expected chunk exists in either top list.
5. Inspect fusion output, then reranker output. Find the stage where the expected chunk falls out.
6. Read the final evidence and gate verdict. Ask whether those exact passages answer the question, not just whether their score is high.
7. Inspect the numbered context produced by `src.prompt.py`.
8. Compare the generated sentences and citations against the evidence and PDF.
9. Check `.env`, Chroma path/collection, embedding model, BM25 path, reranker model, and LM Studio settings.
10. Change extraction/chunking/retrieval/generation only when the failing stage is known. Rebuild only when the indexed content/configuration needs to change.

Do not start by switching the LLM prompt or model: if the right passage is absent from the prompt, that cannot repair retrieval.

## 4. Practical ChromaDB failure scenarios and safe recovery

| Scenario | Likely symptom | How to verify using this code | Safest next action |
|---|---|---|---|
| Persisted Chroma directory/collection does not exist | Dense search may create/open an empty collection and return no results; BM25 may still return candidates. | Check resolved `CHROMA_PATH`, collection `iiitdm_docs`, directory contents, and dense candidate count. | Confirm the configured path and source PDFs first. Back up `storage/` and verify the model/indexing setup before rebuilding. |
| Collection exists but is empty | Dense candidates are empty, while the app may still work through BM25 if other stages work. | Inspect dense count/results and compare a known source page against BM25. A directory existing does not prove the collection has records. | If confirmed empty, preserve a copy, then rebuild from the verified PDFs. |
| Wrong persistence directory | A new empty collection may be opened while the intended index is elsewhere. | `src.config.CHROMA_PATH` resolves `.env` overrides relative to the project root. Print it and inspect both the resolved directory and expected folder. | Correct `.env` to the existing directory. Do not rebuild into the wrong location. |
| Different embedding model | Dimension mismatch may raise an error; same-dimension model mismatch may return meaningless neighbors without a clear error. | Compare `EMBEDDING_MODEL_NAME` used at indexing and query time; inspect dense errors and returned passage quality. | Restore the original model setting or back up and rebuild using one chosen model for both chunks and questions. |
| PDFs were updated but index was not | Current policy text and retrieved evidence disagree or new material is absent. | Compare current PDF pages with chunk text and source metadata in results. The app does not auto-index changed files. | Back up both stores, then rebuild from the complete current source collection. |
| Stale records or duplicate chunks | Old docs remain if chunks are added without a full reset; repeated filenames can collide because chunk IDs are filename/page/chunk based. | Check filenames and IDs. The normal rebuild calls `reset_vectorstore()`, but `index_chunks()` alone does not clear old records. | Back up first. Ensure filenames are unique across categories. Use the full rebuild entry point after confirming its inputs; avoid partial/manual inserts. |
| Missing/incorrect metadata | Retrieval may find the right text but cite “Unknown,” page 0, or a wrong page. | Inspect metadata from `load_pdf_pages()`, `chunk_documents()`, and returned candidates. | Fix source/page assignment before indexing, then rebuild so both stores contain corrected metadata. |
| Similarity is high but passage is not an answer | Dense search returns related vocabulary or topic but misses the requested condition. | Read text and metadata; compare dense, BM25, fused, and reranked lists. | Treat similarity as candidate selection only. Improve the query/source/chunking based on inspection; do not assume a score proves answerability. |
| Collection is reset or overwritten during a failed rebuild | Existing collection has been deleted, or Chroma and BM25 may represent different builds if the script stops partway. `scripts/rebuild_index.py` resets Chroma before adding chunks and writes BM25 after Chroma. | Review the rebuild output and timestamps; compare both stores to the PDF set. The script is not an atomic two-store transaction. | Stop the app, keep the current storage as-is, restore a known-good backup if one exists, or complete a rebuild only after confirming PDFs and models. Back up before another attempt. |
| Local storage moved, deleted, or unavailable | Store load fails, or the configured path resolves to an empty location. | Check `CHROMA_PATH`, `BM25_INDEX_PATH`, actual folders, permissions, and existence of `storage/bm25_index.pkl`. | Restore from backup or correct paths. Rebuild only if source PDFs are present and the required models and packages are available. |
| BM25 and Chroma are from different builds | The two retrievers return different corpora; fusion can combine stale and current chunks or discard IDs that no longer match. | Compare source/page/chunk IDs and build history; the two stores are written sequentially by the indexing script. | Back up and rebuild both stores in the same successful run from one PDF collection. |

### Back up before rebuilding

Rebuilding is not the first diagnostic step. First establish that the source PDFs are correct and that a rebuild is necessary. When it is necessary, close the app and make copies to a new, empty backup location before running the script. For example, in PowerShell from the project root (choose a backup directory name that does not already exist):

```powershell
Copy-Item -LiteralPath .\storage\chroma -Destination .\storage\chroma-backup-before-rebuild -Recurse
Copy-Item -LiteralPath .\storage\bm25_index.pkl -Destination .\storage\bm25_index-before-rebuild.pkl
```

Then confirm the PDFs under `data/raw/`, `.env` paths/model settings, Python dependencies, and local embedding/reranker weights. Only then run:

```powershell
python scripts/rebuild_index.py
```

The rebuild script deliberately deletes the configured Chroma collection before re-indexing it, then overwrites the BM25 pickle. If the run fails after the Chroma reset or before the BM25 save, the stores may be empty or out of sync. Keep the backup until both stores have been checked with known questions. Do not delete the only copy of either store as a troubleshooting shortcut.

## 5. How to evaluate retrieval separately from answers

Retrieval quality asks: **Did the system return the passage that contains the answer?** Answer quality asks: **Did the model state the fact correctly, preserve its conditions, and cite the right source?** A correct retrieval can be followed by a bad answer; a fluent answer can also be a lucky guess after bad retrieval. Evaluate them separately.

The repository includes a small real-source smoke set in [`evaluation/questions.json`](../evaluation/questions.json), including hostel leave, a reworded weekend-leave question, placement registration/eligibility, and a question outside the collection. It is small and is not a benchmark. For each answerable question:

1. Locate the expected fact in the named PDF and confirm the page manually.
2. Run dense and BM25 retrieval separately where possible; record the source, page, chunk ID, rank, and returned text.
3. Check whether the expected passage appears in each list, then in RRF output and the final reranked list.
4. Read the evidence gate's qualifying count and threshold; decide manually whether the evidence actually answers the question.
5. Inspect the prompt context/source labels, generated response, and displayed source/page citations.
6. Record the first failing stage. Do not record an answer as grounded just because the final response shares words with a PDF.

### Retrieval metrics

These are evaluation concepts; the current `evaluation/evaluate_retrieval.py` implements Hit@5, Recall@5, and MRR. It compares retrieved metadata with `expected_source` and `expected_page` in the question file.

- **Precision@k:** of the top `k` returned chunks, how many are relevant. This project does not currently implement Precision@k in `evaluate_retrieval.py`.
- **Recall@k:** of the known relevant chunk(s), how many appear in the top `k`. The script's expected-source/page check acts like a one-target recall check.
- **Hit@k:** whether at least one expected source/page appears in the top `k`. The current script calls this Hit@5.
- **MRR (Mean Reciprocal Rank):** for each question, use `1 / rank` of its first matching chunk (or zero if not found), then average across questions. The script implements this.

Because the current file normally has one expected source/page per answerable question, its `Recall@5` is the same signal as `Hit@5`. A passing retrieval metric still says nothing about whether LM Studio generated a faithful answer. No benchmark score is claimed here.

## 6. Troubleshooting checklist

Work from the source toward the answer, in this order:

1. Confirm the original PDF contains the requested information and the relevant page.
2. Confirm the PDF is in the correct `data/raw/<category>/` folder and was loaded successfully.
3. Confirm extracted and cleaned page text preserves the fact.
4. Confirm the expected chunk and its `source`, `page`, and `chunk_id` exist in the indexes.
5. Inspect dense and BM25 top results separately.
6. Inspect RRF order, then the reranker's top candidates and scores.
7. Inspect the evidence gate count/verdict and judge the selected text yourself.
8. Inspect the numbered prompt context and source labels.
9. Confirm LM Studio is running at the configured endpoint with the intended model; use `python scripts/test_lm_studio.py`.
10. Compare every answer claim and citation with the actual evidence page.
11. Check Chroma/BM25 paths, collection, model consistency, and whether documents changed since indexing.
12. Back up both stores and rebuild only if inspection shows the index is absent, stale, incompatible, or built from incorrect source text.

## 7. Main limitation to remember

ChromaDB is important because it gives this application persistent semantic search over the document chunks. It is only one retrieval store in a hybrid system; BM25 provides a second, keyword-based view. Neither store decides whether a policy passage entails a generated sentence. The answer still depends on correct extraction, retrieval, ranking, evidence selection, prompt construction, local generation, and accurate metadata. The safest way to debug an irrelevant answer is to inspect the evidence at each step, starting with the original PDF, rather than trusting the answer's fluency or changing the LLM first.

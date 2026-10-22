# Streamlit frontend

## Entry point

Run the app with `streamlit run app.py`. The `main()` function renders the header and question form, calls `src.pipeline.answer_question(question)` after submission, and displays the returned result.

## UI behavior

- A text input accepts the question; the **Ask Question** button submits it.
- The answer is shown as text, or a warning is shown when the pipeline refuses.
- The Sources section lists document titles and page numbers returned by the backend.
- Expanders show retrieval counts and retry information, source metadata, the evidence-gate verdict and claim-check counts, and claim statuses.
- LM Studio connection and other runtime errors are displayed in the app.
- Ordinary questions show the local-model answer with a note that it is not grounded in the IIITDM collection. They do not show RAG diagnostics or citations.
- IIITDM/document questions show the RAG answer, source citations, retrieval details, and grounding check.
- Greetings and common app-help questions receive a short description of the assistant and its document coverage without retrieval or generation.

The frontend does not retrieve documents, build prompts, call the language model, or verify claims. Those responsibilities stay in the backend pipeline. The current retrieved-source expander shows citation metadata; it does not display the full passage text.

# Design decisions

- **Keep LM Studio local.** The app uses the existing OpenAI-compatible local endpoint and configured model ID. This keeps generation on the user’s machine and avoids hosted generation APIs.
- **Keep hybrid retrieval.** Dense search helps with paraphrased questions; BM25 helps with exact policy language, dates, and codes. Reciprocal rank fusion combines ranks without assuming the two retrievers’ scores share a scale.
- **Keep the cross-encoder, provisionally.** It reranks the fused candidates and provides the score currently used by the evidence gate. Its quality and the threshold still need measurement against questions grounded in the real documents.
- **Keep page-based chunking.** PDF pages are cleaned, then split into overlapping character chunks. This preserves page citations and keeps chunks small enough for retrieval. It may split a topic across boundaries.
- **Keep refusal checks with clear limitations.** The score gate can stop generation when evidence is weak. The post-generation lexical overlap check is easy to inspect, but it is only a proxy for semantic support.
- **Reject internal reasoning as an answer.** An empty visible LM Studio completion must not be replaced with `reasoning_content`; the generation path now treats it as empty and reports the existing unusable-response error.
- **Do not retrieve for greetings or app-help prompts.** A brief pre-retrieval response describes the app and its document coverage, preventing unrelated high-scoring chunks from turning small talk into a document answer.
- **Separate general chat from document QA.** Ordinary questions use the local LM Studio model without citations; IIITDM and institute-document questions use the retrieval, evidence, and citation pipeline. The UI labels general answers so they are not mistaken for document-grounded results.
- **Keep citation numbering aligned.** Citations correspond to the evidence chunks numbered in the prompt, including multiple chunks from the same page, so source labels remain consistent.
- **No LangSmith tracing.** Optional hosted tracing added an external service and a dependency without being required by the local question-answering flow, so its integration and configuration were removed.

No retrieval-score threshold or model choice was tuned in this cleanup because the available environment lacks the project dependencies needed to run a real retrieval evaluation. The existing persisted indexes were left intact.

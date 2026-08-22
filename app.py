"""
app.py

Streamlit UI for IIITDM Knowledge RAG.

This file contains ONLY presentation logic. All RAG logic lives in
src/pipeline.py — app.py calls answer_question() and renders the result.
Do not add retrieval/generation/verification logic directly in this file.

Layout:
    IIITDM KNOWLEDGE RAG
    ------------------------------------------------
    Question input + [ Ask Question ] button
    ------------------------------------------------
    ANSWER
    ------------------------------------------------
    SOURCES
    ------------------------------------------------
    (expander) Retrieval Details
    (expander) Retrieved Sources
    (expander) Grounding Check
    (expander) Technical Details

Raw internal prompts are NOT exposed by default (see Technical Details
expander note below).

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.pipeline import answer_question, PipelineResult
from src.llm import LMStudioUnavailableError
from src.config import LANGSMITH_ENABLED, LANGSMITH_PROJECT


st.set_page_config(page_title="IIITDM Knowledge RAG", layout="centered")


def render_header() -> None:
    st.title("IIITDM Knowledge RAG")
    st.caption(
        "Document-grounded question answering over official IIITDM Kancheepuram "
        "documents (academics, hostel, placements, institute information)."
    )


def render_question_form() -> str | None:
    """
    Render the question input + Ask Question button.
    """
    st.text_input("Ask your question", key="question_input", placeholder="What are the hostel leave rules?")
    asked = st.button("Ask Question")
    if asked:
        return st.session_state.get("question_input") or None
    return None


def render_answer(result: "PipelineResult") -> None:
    """
    Render the ANSWER section.
    """
    st.subheader("Answer")
    if result.refused:
        st.warning(result.refusal_reason)
    else:
        st.write(result.answer)


def render_sources(result: "PipelineResult") -> None:
    """
    Render the SOURCES section (always visible, not hidden in an expander,
    per the required layout).
    """
    st.subheader("Sources")
    if not result.citations:
        st.write("No sources to show.")
        return
    for c in result.citations:
        st.write(f"[{c.index}] {c.title} - Page {c.page}")


def render_retrieval_details_expander(result: "PipelineResult") -> None:
    with st.expander("🔍 Retrieval Details"):
        st.write("Retrieval method: Hybrid")
        st.write(f"Candidates retrieved: {result.candidates_retrieved}")
        st.write(f"After reranking: {result.candidates_after_rerank}")
        st.write(f"Query reformulation: {'Yes' if result.query_was_reformulated else 'No'}")
        if result.query_was_reformulated:
            st.write(f"Reformulated query: {result.reformulated_query}")
        st.write(f"Detected category: {result.category}")


def render_retrieved_sources_expander(result: "PipelineResult") -> None:
    """
    Shows the full evidence chunks used (text + metadata), not just the
    final citation list — the "show your work" panel for a curious/
    technical reader.
    """
    with st.expander("📚 Retrieved Sources"):
        if not result.citations:
            st.write("No evidence chunks were used.")
            return
        for c in result.citations:
            st.markdown(f"**[{c.index}] {c.title} — Page {c.page}** (`{c.source_filename}`)")


def render_grounding_check_expander(result: "PipelineResult") -> None:
    with st.expander("🛡 Grounding Check"):
        verdict = result.evidence_verdict
        st.write(f"Evidence sufficient: {'Yes' if verdict.status.value == 'SUFFICIENT' else 'No'}")
        st.write(
            f"Qualifying chunks: {verdict.num_qualifying_chunks} "
            f"(threshold {verdict.threshold_used}, minimum {verdict.min_chunks_required})"
        )
        vr = result.verification_result
        if vr is not None:
            st.write(f"Supported claims: {vr.supported_count}/{vr.total_count}")
            st.write(f"Unsupported claims: {vr.total_count - vr.supported_count}")
        else:
            st.write("No claim verification was run (answer was refused before generation).")


def render_technical_details_expander(result: "PipelineResult") -> None:
    """
    Lower-level debugging info for development use. Raw internal prompts
    are intentionally NOT shown here, per project requirements — this is
    diagnostics, not a prompt debugger.
    """
    with st.expander("⚙️ Technical Details"):
        st.write(f"Category detected: {result.category}")
        st.write(f"Refused: {result.refused}")
        vr = result.verification_result
        if vr is not None:
            for claim in vr.claims:
                st.write(f"- [{claim.status.value}] {claim.claim_text}")

        # LangSmith is optional observability only — never required for the
        # app to function. This just reflects config.py's view of whether
        # tracing is currently active.
        st.divider()
        if LANGSMITH_ENABLED:
            st.write(f"LangSmith tracing: ON (project: {LANGSMITH_PROJECT})")
        else:
            st.write("LangSmith tracing: OFF")


def main() -> None:
    render_header()
    question = render_question_form()

    if question:
        try:
            with st.spinner("Retrieving and generating..."):
                result = answer_question(question)
        except LMStudioUnavailableError:
            st.error("LM Studio server is not available. Start the local server and try again.")
            return
        except RuntimeError as exc:
            st.error(str(exc))
            return

        render_answer(result)
        render_sources(result)
        render_retrieval_details_expander(result)
        render_retrieved_sources_expander(result)
        render_grounding_check_expander(result)
        render_technical_details_expander(result)


if __name__ == "__main__":
    main()

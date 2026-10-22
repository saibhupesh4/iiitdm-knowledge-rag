"""
query_reformulation.py

Runs only when evidence_gate.py returns INSUFFICIENT. Rewrites the user's
original query to make important keywords/entities explicit, then hands
control back to pipeline.py for a single bounded retry of hybrid retrieval
+ reranking.

Bounded by config.MAX_QUERY_REFORMULATION_RETRIES (default 1) — this
module must never be called in an unbounded loop. The retry-count
bookkeeping lives in pipeline.py, not here; this module only produces one
reformulated query per call.

Example:
    Original:      "What about hostel leave?"
    Reformulated:  "What is the IIITDM hostel leave application procedure,
                     required notice period, and leave form?"

Input:  original_query (str), detected_category (str, from
        query_understanding.py)
Output: reformulated_query (str)

Implementation note: reformulation itself can be done with a local LLM
call (via llm.py) or with simple rule-based expansion using the detected
category — start with whichever is simpler to get right, and document
the choice here once implemented.
"""

from __future__ import annotations


# Rule-based expansion chosen over an LLM call: it is simple, fully
# explainable, free of an extra network round-trip to LM Studio, and
# matches the project's "no unnecessary complexity" rule. Each category
# expands the query with domain terms that are likely to appear verbatim
# in the official documents, helping both BM25 (lexical) and dense
# retrieval find better candidates on the retry.
CATEGORY_EXPANSIONS: dict[str, str] = {
    "academic": (
        "IIITDM Kancheepuram academic regulations curriculum syllabus "
        "attendance examination credit requirements"
    ),
    "hostel": (
        "IIITDM Kancheepuram hostel rules regulations leave application "
        "warden mess procedure"
    ),
    "placement": (
        "IIITDM Kancheepuram placement policy internship eligibility "
        "recruitment procedure"
    ),
    "institute": (
        "IIITDM Kancheepuram institute information administration "
        "department facilities"
    ),
    "unknown": "IIITDM Kancheepuram official policy document",
}


def reformulate_query(original_query: str, category: str = "unknown") -> str:
    """
    Produce a reformulated version of original_query that makes intent and
    key entities explicit, to give retrieval a second, better chance.

    Category terms are appended to give lexical retrieval more document
    vocabulary to match on its single bounded retry.
    """
    expansion = CATEGORY_EXPANSIONS.get(category, CATEGORY_EXPANSIONS["unknown"])
    return f"{original_query.strip()} {expansion}"


if __name__ == "__main__":
    print(reformulate_query("What about hostel leave?", category="hostel"))

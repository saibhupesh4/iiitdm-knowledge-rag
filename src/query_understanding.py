"""
query_understanding.py

Lightweight query understanding / category detection, run BEFORE retrieval.

Deliberately simple and explainable (rule-based / keyword matching to
start) rather than a trained classifier or an agent — per the locked
architecture, this must stay a small, inspectable component.

Categories (see config.QUERY_CATEGORIES):
    academic | hostel | placement | institute | unknown

The detected category can optionally influence retrieval (e.g. metadata
filtering or a ranking boost for chunks whose category matches) — that
wiring happens in hybrid_retrieval.py / pipeline.py, not here.

Input:  a user question (str)
Output: a category (str), one of config.QUERY_CATEGORIES

Example:
    "What is the hostel leave procedure?" -> "hostel"
"""

from __future__ import annotations

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator

from src.config import QUERY_CATEGORIES

# TODO: refine these keyword lists against real IIITDM document vocabulary
# once documents are available. Keep this mapping simple and easy to read —
# that is the point of this module.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "academic": ["curriculum", "course", "credit", "attendance", "exam", "syllabus", "grade", "cgpa"],
    "hostel": ["hostel", "leave", "warden", "mess", "room", "curfew"],
    "placement": ["placement", "internship", "company", "package", "recruit", "offer"],
    "institute": ["institute", "iiitdm", "campus", "director", "department", "facility"],
}


@traceable(name="query_understanding", run_type="chain")
def detect_category(question: str) -> str:
    """
    Return the best-matching category for a question, or "unknown" if no
    keywords match.

    Tie-break rule: if two or more categories have the same (non-zero)
    top match count, return "unknown" rather than guessing, since a
    confident-but-wrong category could bias retrieval.
    """
    lowered = question.lower()

    match_counts: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lowered)
        if count > 0:
            match_counts[category] = count

    if not match_counts:
        return "unknown"

    best_count = max(match_counts.values())
    best_categories = [c for c, n in match_counts.items() if n == best_count]

    if len(best_categories) > 1:
        return "unknown"

    return best_categories[0]


if __name__ == "__main__":
    print(detect_category("What is the hostel leave procedure?"))

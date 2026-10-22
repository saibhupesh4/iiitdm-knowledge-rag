"""
query_understanding.py

Lightweight query understanding / category detection, run BEFORE retrieval.

Deliberately simple and explainable (rule-based / keyword matching to
start) rather than a trained classifier or an agent — per the locked
architecture, this must stay a small, inspectable component.

Categories (see config.QUERY_CATEGORIES):
    academic | hostel | placement | institute | unknown

The router sends ordinary questions to general local chat and IIITDM or
institute-document topics to RAG. Category detection supplies terms for
the bounded retry in the RAG path.

Input:  a user question (str)
Output: a category (str), one of config.QUERY_CATEGORIES

Example:
    "What is the hostel leave procedure?" -> "hostel"
"""

from __future__ import annotations

import re

from src.config import QUERY_CATEGORIES

# These categories are used only to choose fixed terms for a weak-evidence
# retry; they do not filter the initial retrieval results.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "academic": ["curriculum", "course", "credit", "attendance", "exam", "syllabus", "grade", "cgpa"],
    "hostel": ["hostel", "leave", "warden", "mess", "room", "curfew"],
    "placement": ["placement", "internship", "company", "package", "recruit", "offer"],
    "institute": ["institute", "iiitdm", "campus", "director", "department", "facility"],
}

_APP_HELP_INPUTS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "who are you",
    "what are you",
    "what can you do",
    "what is your purpose",
    "how can you help",
    "tell me about yourself",
    "what is this app",
    "help",
}

_DOCUMENT_TOPICS = {
    keyword
    for keywords in CATEGORY_KEYWORDS.values()
    for keyword in keywords
} | {
    "academic calendar", "admission", "anti ragging", "btech", "complaint",
    "fee", "fees", "faculty", "handbook", "harassment", "icc", "institute",
    "kancheepuram", "mtech", "ordinance", "phd", "policy", "regulation",
    "rules", "scholarship", "semester", "tuition",
}


def is_app_help_input(question: str) -> bool:
    """Recognize greetings and app-help prompts that need no document search."""
    normalized = re.sub(r"[^\w\s]", "", question.casefold()).strip()
    return normalized in _APP_HELP_INPUTS


def is_document_question(question: str) -> bool:
    """Route IIITDM and institute-document topics through RAG."""
    normalized = re.sub(r"[^\w\s]", " ", question.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(
        re.search(rf"\b{re.escape(topic)}\b", normalized)
        for topic in _DOCUMENT_TOPICS
    )


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

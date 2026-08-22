"""
verification.py

Claim/evidence verification for the IIITDM Knowledge RAG system.

The verifier checks whether factual statements in the generated answer
are supported by the retrieved evidence.

Verification flow:

    Draft answer
        ↓
    Remove headings / formatting
        ↓
    Extract factual claims
        ↓
    Read cited [Source N] references
        ↓
    Compare claim with cited evidence
        ↓
    Lexical support scoring
        ↓
    VerificationResult

This is a deterministic grounding check. It reduces unsupported claims
but does not guarantee complete factual correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover

    def traceable(*args, **kwargs):

        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


from src.reranker import RerankedCandidate


# ============================================================
# Configuration
# ============================================================

SUPPORT_OVERLAP_THRESHOLD = 0.50

MIN_CONTENT_WORDS = 3

MIN_SUPPORT_RATIO = 0.80


# ============================================================
# Regular expressions
# ============================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"
)

_SINGLE_SOURCE_RE = re.compile(
    r"\[Source\s+(\d+)\]",
    re.IGNORECASE,
)

_ANY_CITATION_RE = re.compile(
    r"\[Source\s+\d+\]",
    re.IGNORECASE,
)

_BULLET_RE = re.compile(
    r"^\s*(?:[-•*]|\d+[.)])\s*"
)

_MARKDOWN_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s+(.+?)\s*$"
)


# ============================================================
# Stopwords
# ============================================================

_STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "if", "then", "else",
    "of", "to", "in", "on",
    "at", "by", "for", "from",
    "with", "without", "as",
    "into", "about",

    "is", "are", "was", "were",
    "be", "been", "being",

    "do", "does", "did",

    "has", "have", "had",

    "can", "could", "will",
    "would", "shall", "should",
    "may", "might", "must",

    "i", "we", "you", "he",
    "she", "they", "it",

    "me", "us", "him", "her",
    "them",

    "my", "our", "your",
    "his", "their", "its",

    "this", "that", "these",
    "those",

    "not", "no", "yes",

    "also", "very", "more",
    "most", "some", "any",
    "all", "each", "every",
}


# ============================================================
# Enums
# ============================================================

class ClaimSupport(str, Enum):

    SUPPORTED = "SUPPORTED"

    UNSUPPORTED = "UNSUPPORTED"


# ============================================================
# Data classes
# ============================================================

@dataclass
class ClaimCheck:

    claim_text: str

    status: ClaimSupport

    support_score: float

    supporting_source_indices: list[int]


@dataclass
class VerificationResult:

    claims: list[ClaimCheck]

    @property
    def supported_count(self) -> int:

        return sum(
            claim.status == ClaimSupport.SUPPORTED
            for claim in self.claims
        )

    @property
    def unsupported_count(self) -> int:

        return sum(
            claim.status == ClaimSupport.UNSUPPORTED
            for claim in self.claims
        )

    @property
    def total_count(self) -> int:

        return len(self.claims)

    @property
    def support_ratio(self) -> float:

        if self.total_count == 0:
            return 0.0

        return round(
            self.supported_count / self.total_count,
            4,
        )

    @property
    def all_supported(self) -> bool:

        return (
            self.total_count > 0
            and self.supported_count == self.total_count
        )

    @property
    def sufficiently_supported(self) -> bool:

        if self.total_count == 0:
            return False

        return (
            self.support_ratio >= MIN_SUPPORT_RATIO
        )


# ============================================================
# Text utilities
# ============================================================

def _content_words(text: str) -> set[str]:
    """
    Extract meaningful lowercase words.
    """

    tokens = _TOKEN_RE.findall(
        text.lower()
    )

    return {
        token
        for token in tokens
        if token not in _STOPWORDS
        and len(token) > 1
    }


def _extract_citation_indices(
    text: str,
) -> list[int]:
    """
    Extract zero-based evidence indices.

    Example:

        [Source 1][Source 3]

    becomes:

        [0, 2]
    """

    numbers = _SINGLE_SOURCE_RE.findall(text)

    indices: list[int] = []

    for number in numbers:

        index = int(number) - 1

        if index >= 0 and index not in indices:
            indices.append(index)

    return indices


def _remove_citations(text: str) -> str:

    return _ANY_CITATION_RE.sub(
        "",
        text,
    )


def _clean_claim_text(text: str) -> str:

    text = _remove_citations(text)

    text = _BULLET_RE.sub(
        "",
        text,
    )

    text = text.strip()

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    return text.strip()


def _is_heading(
    raw_text: str,
    cleaned_text: str,
) -> bool:
    """
    Detect formatting-only headings.

    Examples:

        ## Documentation Requirements
        Special Cases
        Approval Requirements
    """

    raw_text = raw_text.strip()
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        return True

    if _MARKDOWN_HEADING_RE.match(raw_text):
        return True

    if raw_text.endswith(":"):
        return True

    if cleaned_text.endswith(":"):
        return True

    words = _content_words(cleaned_text)

    if len(words) <= 3:

        if not re.search(
            r"[.!?]$",
            cleaned_text,
        ):
            return True

    return False


# ============================================================
# Claim extraction
# ============================================================

def extract_claims(
    draft_answer: str | None,
) -> list[tuple[str, list[int]]]:
    """
    Extract factual claims and their cited source indices.

    Returns:

        [
            (
                "Students must submit leave requests in advance.",
                [0, 1],
            )
        ]
    """

    if not draft_answer:
        return []

    text = draft_answer.strip()

    if not text:
        return []

    lines = text.splitlines()

    claims: list[
        tuple[str, list[int]]
    ] = []

    for line in lines:

        raw_line = line.strip()

        if not raw_line:
            continue

        cleaned_line = _clean_claim_text(
            raw_line
        )

        if _is_heading(
            raw_line,
            cleaned_line,
        ):
            continue

        sentence_parts = _SENTENCE_SPLIT_RE.split(
            raw_line
        )

        for part in sentence_parts:

            claim = _clean_claim_text(
                part
            )

            if not claim:
                continue

            if _is_heading(
                part,
                claim,
            ):
                continue

            words = _content_words(
                claim
            )

            if len(words) < MIN_CONTENT_WORDS:
                continue

            cited_sources = (
                _extract_citation_indices(
                    part
                )
            )

            claims.append(
                (
                    claim,
                    cited_sources,
                )
            )

    return claims


# ============================================================
# Support scoring
# ============================================================

def _calculate_support_score(
    claim_words: set[str],
    evidence_words: set[str],
) -> float:
    """
    Calculate fraction of meaningful claim words
    found in an evidence chunk.
    """

    if not claim_words:
        return 0.0

    overlap = (
        claim_words
        & evidence_words
    )

    return len(overlap) / len(claim_words)


# ============================================================
# Single claim verification
# ============================================================

def verify_claim(
    claim: str,
    cited_source_indices: list[int],
    evidence: list[RerankedCandidate] | None,
) -> ClaimCheck:
    """
    Verify one factual claim.

    If the answer cites sources, verification checks those sources.

    If no citation exists, all retrieved evidence is checked.
    """

    if not evidence:

        return ClaimCheck(
            claim_text=claim,
            status=ClaimSupport.UNSUPPORTED,
            support_score=0.0,
            supporting_source_indices=[],
        )

    claim_words = _content_words(
        claim
    )

    if not claim_words:

        return ClaimCheck(
            claim_text=claim,
            status=ClaimSupport.UNSUPPORTED,
            support_score=0.0,
            supporting_source_indices=[],
        )

    # --------------------------------------------------------
    # Decide which evidence chunks to check
    # --------------------------------------------------------

    if cited_source_indices:

        candidate_indices = [
            index
            for index in cited_source_indices
            if 0 <= index < len(evidence)
        ]

    else:

        candidate_indices = list(
            range(len(evidence))
        )

    if not candidate_indices:

        return ClaimCheck(
            claim_text=claim,
            status=ClaimSupport.UNSUPPORTED,
            support_score=0.0,
            supporting_source_indices=[],
        )

    # --------------------------------------------------------
    # Compare claim with evidence
    # --------------------------------------------------------

    best_score = 0.0

    supporting_indices: list[int] = []

    for index in candidate_indices:

        chunk = evidence[index]

        evidence_words = _content_words(
            chunk.text
        )

        if not evidence_words:
            continue

        score = _calculate_support_score(
            claim_words,
            evidence_words,
        )

        best_score = max(
            best_score,
            score,
        )

        if score >= SUPPORT_OVERLAP_THRESHOLD:

            supporting_indices.append(
                index
            )

    status = (
        ClaimSupport.SUPPORTED
        if supporting_indices
        else ClaimSupport.UNSUPPORTED
    )

    return ClaimCheck(
        claim_text=claim,
        status=status,
        support_score=round(
            best_score,
            4,
        ),
        supporting_source_indices=supporting_indices,
    )


# ============================================================
# Answer verification
# ============================================================

@traceable(
    name="verification",
    run_type="chain",
)
def verify_answer(
    draft_answer: str | None,
    evidence: list[RerankedCandidate] | None,
) -> VerificationResult:
    """
    Verify all factual claims in a generated answer.
    """

    extracted_claims = extract_claims(
        draft_answer
    )

    checks: list[ClaimCheck] = []

    for claim, cited_indices in extracted_claims:

        check = verify_claim(
            claim=claim,
            cited_source_indices=cited_indices,
            evidence=evidence,
        )

        checks.append(check)

    return VerificationResult(
        claims=checks
    )


# ============================================================
# Local tests
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "IIITDM Knowledge RAG - Verification Tests"
    )
    print("=" * 60)

    answer = """
# Hostel Leave Rules

## Advance Application Process

Students must submit a leave request in advance
through the prescribed format [Source 1].

Leave applications should be sent at least two days
in advance [Source 2].

## Approval Requirements

Leave requires prior approval [Source 1][Source 2].
"""

    extracted = extract_claims(answer)

    print("\nExtracted claims:")

    for claim, sources in extracted:

        print("-" * 60)

        print(
            "Claim:",
            claim,
        )

        print(
            "Cited sources:",
            [
                index + 1
                for index in sources
            ],
        )

    print()
    print("=" * 60)
    print(
        "VERIFICATION TESTS COMPLETED"
    )
    print("=" * 60)
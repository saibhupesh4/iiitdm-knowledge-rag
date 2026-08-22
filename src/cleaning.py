"""
cleaning.py

STEP 2 of the pipeline:

RAW PDF PAGE TEXT
        ↓
REMOVE PDF ARTIFACTS
        ↓
NORMALIZE WHITESPACE
        ↓
REMOVE EMPTY PAGES
        ↓
DETECT SECTION HEADING
        ↓
CLEANED DOCUMENTS + METADATA

Responsible for:
    - Removing empty pages
    - Normalizing excessive whitespace
    - Removing obvious PDF extraction artifacts
    - Preserving meaningful headings
    - Detecting section headings where possible

This module must NOT aggressively rewrite text.

Numbers, dates, percentages, course codes, rules, and policy terms
must remain unchanged.

Input:
    List of raw LangChain Document objects from ingestion.py

Output:
    List of cleaned LangChain Document objects.

Metadata is preserved and enriched with:

    {
        "source": "...",
        "page": 3,
        "category": "hostel",
        "title": "...",
        "section": "Attendance Requirements"
    }
"""

from __future__ import annotations

import re

from langchain_core.documents import Document


# =============================================================
# Regex patterns
# =============================================================

# Remove control characters but preserve:
# \n = newline
# \t = tab
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
)

# Multiple spaces or tabs
_MULTI_SPACE_RE = re.compile(
    r"[ \t]+"
)

# Three or more newlines -> maximum two
_MULTI_NEWLINE_RE = re.compile(
    r"\n{3,}"
)

# Standalone page number
# Examples:
# 1
# 17
# 203
_STANDALONE_PAGE_NUMBER_RE = re.compile(
    r"^\s*\d{1,4}\s*$"
)

# Numbered headings
#
# Examples:
# 3 Attendance Requirements
# 3. Leave Rules
# 3.2 Attendance Requirements
# 4.1.2 Examination Rules
_NUMBERED_HEADING_RE = re.compile(
    r"""
    ^\s*
    \d+(?:\.\d+)*
    [.)]?
    \s+
    ([A-Za-z][A-Za-z0-9 ,&/'()\-]{2,100})
    \s*$
    """,
    re.VERBOSE,
)

# ALL CAPS heading
#
# Examples:
# HOSTEL RULES
# ATTENDANCE REQUIREMENTS
# ACADEMIC REGULATIONS 2025
_ALL_CAPS_HEADING_RE = re.compile(
    r"^[A-Z0-9][A-Z0-9 &,/'()\-]{4,100}$"
)


# =============================================================
# Whitespace normalization
# =============================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive whitespace without changing the meaning
    of the text.

    Preserves:
        - line structure
        - numbers
        - dates
        - percentages
        - course codes
        - punctuation
        - casing
    """

    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse repeated spaces/tabs
    text = _MULTI_SPACE_RE.sub(" ", text)

    # Remove leading/trailing spaces from each line
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Maximum two consecutive newlines
    text = _MULTI_NEWLINE_RE.sub(
        "\n\n",
        text
    )

    return text.strip()


# =============================================================
# PDF artifact removal
# =============================================================

def remove_pdf_artifacts(text: str) -> str:
    """
    Remove obvious PDF extraction artifacts.

    Conservative cleaning only.

    We intentionally do NOT remove:
        - numbers inside sentences
        - dates
        - percentages
        - course codes
        - rule numbers attached to headings
        - meaningful text
    """

    # Remove unwanted control characters
    text = _CONTROL_CHARS_RE.sub(
        " ",
        text
    )

    kept_lines = []

    for line in text.split("\n"):

        stripped = line.strip()

        # Remove standalone page number
        if _STANDALONE_PAGE_NUMBER_RE.match(
            stripped
        ):
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines)


# =============================================================
# Empty page detection
# =============================================================

def is_empty_page(text: str) -> bool:
    """
    Return True if the page contains no meaningful content.
    """

    stripped = text.strip()

    if not stripped:
        return True

    # Require at least one alphabetic character.
    # This prevents pages containing only:
    # page numbers, punctuation, symbols, etc.
    if not re.search(
        r"[A-Za-z]",
        stripped
    ):
        return True

    return False


# =============================================================
# Section heading detection
# =============================================================

def detect_section_heading(
    text: str
) -> str | None:
    """
    Try to detect a meaningful section heading near the
    beginning of a page.

    Examples:

        "3. Attendance Requirements"
            ->
        "Attendance Requirements"

        "HOSTEL LEAVE RULES"
            ->
        "Hostel Leave Rules"

    Returns:
        Heading string or None.
    """

    # Usually headings occur near the beginning.
    # Looking at first 20 non-empty lines is sufficient.
    checked = 0

    for line in text.split("\n"):

        stripped = line.strip()

        if not stripped:
            continue

        checked += 1

        if checked > 20:
            break

        # Numbered heading
        match = _NUMBERED_HEADING_RE.match(
            stripped
        )

        if match:
            return match.group(1).strip()

        # ALL CAPS heading
        if (
            _ALL_CAPS_HEADING_RE.match(stripped)
            and len(stripped.split()) <= 12
        ):
            return stripped.title()

    return None


# =============================================================
# Main cleaning function
# =============================================================

def clean_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Apply the complete cleaning pipeline.

    Pipeline:

        Raw Document
             ↓
        Remove artifacts
             ↓
        Normalize whitespace
             ↓
        Remove empty pages
             ↓
        Detect section
             ↓
        Cleaned Document

    Metadata from ingestion.py is preserved.
    """

    cleaned: list[Document] = []

    removed_empty = 0

    for doc in documents:

        # Step 1: remove extraction artifacts
        text = remove_pdf_artifacts(
            doc.page_content
        )

        # Step 2: normalize whitespace
        text = normalize_whitespace(
            text
        )

        # Step 3: remove empty pages
        if is_empty_page(text):

            removed_empty += 1

            continue

        # Preserve original metadata
        metadata = dict(
            doc.metadata
        )

        # Step 4: detect section
        section = detect_section_heading(
            text
        )

        if section is not None:
            metadata["section"] = section
        else:
            metadata["section"] = None

        # Create cleaned document
        cleaned.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    print(
        f"[cleaning] "
        f"Input pages: {len(documents)}"
    )

    print(
        f"[cleaning] "
        f"Empty pages removed: {removed_empty}"
    )

    print(
        f"[cleaning] "
        f"Cleaned pages: {len(cleaned)}"
    )

    return cleaned


# =============================================================
# Standalone test
# =============================================================

if __name__ == "__main__":

    from src.ingestion import (
        load_all_documents,
    )

    print(
        "IIITDM Knowledge RAG - Cleaning Test"
    )

    print("=" * 60)

    print("\nLoading documents...\n")

    raw_docs = load_all_documents()

    print()

    print("=" * 60)

    print("\nCleaning documents...\n")

    cleaned_docs = clean_documents(
        raw_docs
    )

    print()

    print("=" * 60)

    print(
        "\nFINAL RESULT"
    )

    print(
        f"Raw pages: {len(raw_docs)}"
    )

    print(
        f"Cleaned pages: {len(cleaned_docs)}"
    )

    if cleaned_docs:

        print("\nExample metadata:")

        print(
            cleaned_docs[0].metadata
        )

        print("\nExample cleaned text:")

        preview = (
            cleaned_docs[0]
            .page_content[:500]
        )

        print(preview)

    else:

        print(
            "\nNo cleaned documents produced."
        )
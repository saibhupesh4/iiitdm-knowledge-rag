"""
ingestion.py

STEP 1 of the pipeline:

OFFICIAL IIITDM PDFs
        ↓
PDF DISCOVERY
        ↓
PYPDFLOADER
        ↓
LANGCHAIN DOCUMENTS

Responsible for:
    - Walking data/raw/<category>/ folders
    - Discovering PDF files
    - Loading each PDF page by page
    - Attaching consistent metadata

Explicitly NOT responsible for:
    - Cleaning text
    - Chunking
    - Embedding
    - Indexing

Output:
    One LangChain Document per PDF page.

Example metadata:
    {
        "source": "hostel_rules_2025_26.pdf",
        "page": 3,
        "category": "hostel",
        "title": "Hostel Rules 2025 26"
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import RAW_DATA_PATH


# =============================================================
# Constants
# =============================================================

CATEGORIES = [
    "academic",
    "hostel",
    "placement",
    "institute",
]


# =============================================================
# Source document record
# =============================================================

@dataclass
class SourceDocument:
    """
    Represents a PDF discovered in the raw data directory.
    """

    path: Path
    category: str
    title: str


# =============================================================
# Filename -> readable title
# =============================================================

def _title_from_filename(path: Path) -> str:
    """
    Convert a filename into a readable title.

    Example:
        hostel_rules_2025_26.pdf
            ->
        Hostel Rules 2025 26
    """

    stem = path.stem

    stem = stem.replace("_", " ")
    stem = stem.replace("-", " ")

    return " ".join(
        word.capitalize()
        for word in stem.split()
    )


# =============================================================
# PDF discovery
# =============================================================

def discover_pdfs(
    raw_data_dir: Path = RAW_DATA_PATH,
) -> list[SourceDocument]:
    """
    Discover all PDFs inside:

        data/raw/
            academic/
            hostel/
            placement/
            institute/

    Returns:
        List of SourceDocument objects.
    """

    discovered: list[SourceDocument] = []

    for category in CATEGORIES:

        category_dir = raw_data_dir / category

        if not category_dir.exists():
            print(
                f"[ingestion] Category folder not found: "
                f"{category_dir}"
            )
            continue

        pdf_files = sorted(
            category_dir.glob("*.pdf")
        )

        for pdf_path in pdf_files:

            discovered.append(
                SourceDocument(
                    path=pdf_path,
                    category=category,
                    title=_title_from_filename(
                        pdf_path
                    ),
                )
            )

    return discovered


# =============================================================
# Single PDF loading
# =============================================================

def load_pdf_pages(
    source_doc: SourceDocument,
) -> list[Document]:
    """
    Load one PDF.

    Returns:
        One LangChain Document per PDF page.

    Metadata added to every page:
        source
        page
        category
        title
    """

    loader = PyPDFLoader(
        str(source_doc.path)
    )

    raw_pages = loader.load()

    pages: list[Document] = []

    for index, page_doc in enumerate(
        raw_pages,
        start=1,
    ):

        pages.append(
            Document(
                page_content=page_doc.page_content,

                metadata={
                    "source": source_doc.path.name,

                    # 1-indexed page number.
                    # This is what will be shown
                    # later in citations.
                    "page": index,

                    "category": source_doc.category,

                    "title": source_doc.title,
                },
            )
        )

    return pages


# =============================================================
# Load all PDFs
# =============================================================

def load_all_documents(
    raw_data_dir: Path = RAW_DATA_PATH,
) -> list[Document]:
    """
    Discover and load every PDF.

    If one PDF fails, log the error and continue
    processing the remaining PDFs.

    Returns:
        List of LangChain Document objects,
        one per PDF page.
    """

    source_documents = discover_pdfs(
        raw_data_dir
    )

    if not source_documents:

        print(
            "[ingestion] WARNING: "
            "No PDF files found."
        )

        return []

    all_pages: list[Document] = []

    total_pdfs = len(
        source_documents
    )

    print(
        f"[ingestion] Found "
        f"{total_pdfs} PDF files."
    )

    for index, source_doc in enumerate(
        source_documents,
        start=1,
    ):

        print(
            f"[ingestion] "
            f"[{index}/{total_pdfs}] "
            f"Loading: "
            f"{source_doc.path.name}"
        )

        try:

            pages = load_pdf_pages(
                source_doc
            )

            all_pages.extend(
                pages
            )

            print(
                f"    -> Loaded "
                f"{len(pages)} pages"
            )

        except Exception as exc:

            print(
                f"    -> WARNING: "
                f"Failed to load "
                f"{source_doc.path.name}"
            )

            print(
                f"       Reason: {exc}"
            )

    print()

    print(
        f"[ingestion] Total pages loaded: "
        f"{len(all_pages)}"
    )

    return all_pages


# =============================================================
# Standalone test
# =============================================================

if __name__ == "__main__":

    print(
        "IIITDM Knowledge RAG - "
        "Ingestion Test"
    )

    print("=" * 60)

    print(
        f"Raw data directory:\n"
        f"{RAW_DATA_PATH}"
    )

    print()

    documents = load_all_documents()

    print()
    print("=" * 60)

    print(
        f"FINAL RESULT: "
        f"{len(documents)} pages loaded"
    )

    if documents:

        print()

        print(
            "Example metadata:"
        )

        print(
            documents[0].metadata
        )

        print()

        preview = (
            documents[0]
            .page_content[:300]
            .replace("\n", " ")
        )

        print(
            "Example text:"
        )

        print(preview)

    else:

        print(
            "No documents were loaded."
        )
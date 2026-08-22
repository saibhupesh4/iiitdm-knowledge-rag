# Data Directory

## Where to place documents

Place official IIITDM Kancheepuram PDFs in the matching category folder
under `data/raw/`:

```
data/raw/
    academic/     <- curriculum, academic regulations, academic calendar
    hostel/       <- hostel rules, hostel leave rules
    placement/    <- placement brochure, placement procedure
    institute/    <- institute/student information, general notices
```

Only official, institute-issued documents belong here — do not add
material scraped from unrelated websites or blogs.

Start with approximately 10–20 high-quality documents. More can be added
later; re-run `python scripts/rebuild_index.py` after adding or changing
any file so the vector store and BM25 index stay in sync with `data/raw/`.

## `data/processed/`

Intermediate cleaned/chunked output (if the pipeline is configured to
cache it) lives here. This directory is derived entirely from
`data/raw/` — nothing here should be hand-edited, and it's safe to
delete and regenerate via `scripts/ingest.py` / `scripts/rebuild_index.py`.

## Expected chunk metadata

Every chunk produced by the pipeline should carry:

| Field      | Meaning                                              | Example                            |
|------------|-------------------------------------------------------|-------------------------------------|
| `source`   | Original PDF filename                                | `hostel_rules_2025_26.pdf`          |
| `page`     | Page number within that PDF (1-indexed)              | `3`                                  |
| `category` | Folder it was placed in                              | `hostel`                             |
| `title`    | Human-readable document title                        | `Hostel Rules and Regulations`      |
| `section`  | Detected section/heading, where available            | `Leave Rules`                        |

Page numbers are essential — the application shows page-level citations
for every answer, and a citation is never shown unless it maps back to
an actual retrieved chunk with this metadata attached.

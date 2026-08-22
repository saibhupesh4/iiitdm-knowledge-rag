# Storage Directory

This directory holds locally persisted indexes so the application does
not need to re-embed or re-index documents on every startup.

```
storage/
    chroma/            <- persisted Chroma vector store (embeddings + chunk text + metadata)
    bm25_index.pkl      <- persisted BM25 index (created by scripts/rebuild_index.py)
```

Nothing in this directory should be hand-edited. It is entirely derived
from `data/raw/` via:

```bash
python scripts/rebuild_index.py
```

Run that script whenever documents in `data/raw/` are added, removed, or
changed. `storage/chroma/` contents are git-ignored (see `.gitignore`) —
only the folder structure (`.gitkeep`) is tracked, since the actual index
is large, machine-specific, and fully rebuildable.

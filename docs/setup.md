# Setup

## 1. Python environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Environment variables

```bash
cp .env.example .env
```

Then edit `.env`. At minimum, `LM_STUDIO_MODEL` must match the model
name shown in LM Studio's "Local Server" tab (see `docs/lm_studio.md`).
No `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` is required
anywhere in this project.

## 3. LM Studio

See `docs/lm_studio.md` for full setup. Quick version:

1. Install LM Studio and load a chat model.
2. Start its local server (default `http://localhost:1234/v1`).
3. Verify connectivity:
   ```bash
   python scripts/test_lm_studio.py
   ```

## 4. Add documents

Place official IIITDM PDFs under `data/raw/<category>/` — see
`data/README.md` for the expected folder layout and metadata.

## 5. Build the indexes

```bash
python scripts/rebuild_index.py
```

This builds and persists both the Chroma vector index and the BM25
index under `storage/`. Re-run it whenever documents change.

## 6. Run the app

```bash
streamlit run app.py
```

## 7. Run evaluation (optional, once documents + indexes exist)

```bash
python evaluation/evaluate_retrieval.py
python evaluation/evaluate_generation.py
python evaluation/evaluate_grounding.py
python evaluation/ablation.py
```

# Local setup

## Python environment

From the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `LM_STUDIO_MODEL` to the exact model ID shown by the
LM Studio local server. The sample value is only a placeholder.

## LM Studio

Load a local chat/instruct model and start the server. The default endpoint
is `http://127.0.0.1:1234/v1`. Check connectivity with:

```powershell
python scripts/test_lm_studio.py
```

## Documents and indexes

Official PDFs belong under `data/raw/<category>/`. The existing indexes in
`storage/` are loaded by the app. Rebuild them only after changing the PDFs
or indexing settings; rebuilding replaces the current persisted indexes:

```powershell
python scripts/rebuild_index.py
```

## Start the app

```powershell
streamlit run app.py
```

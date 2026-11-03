## Information

- Name: Nguyễn Hoàng Long
- Student ID: 25127090

## Model and Processes

The selected model is `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, loaded by llama.cpp and served through its OpenAI-compatible API. Source repository metadata: [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF).

Run three local processes:

1. llama.cpp model server: `http://127.0.0.1:1234/`
2. FastAPI JSON API: `http://127.0.0.1:8000/`
3. Static frontend: `http://127.0.0.1:3000/`

## Setup

Prerequisites: Python 3.11+, a running llama.cpp server with the model loaded, and PowerShell on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Verify `.env` contains the local server settings:

```dotenv
LLM_BACKEND=llamacpp
LLAMACPP_BASE_URL=http://127.0.0.1:1234
LLAMACPP_MODEL=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Start the API from the project root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal, serve the frontend:

```powershell
python -m http.server 3000 --directory frontend --bind 127.0.0.1
```

The frontend defaults to API base URL `http://127.0.0.1:8000` and has an editable API URL field. CORS allows only `http://127.0.0.1:3000` and `http://localhost:3000` by default; configure `FRONTEND_ORIGINS` as an explicit comma-separated list if needed.

The frontend's **API activity** panel shows request method/path, pending/completed/error state, status, elapsed time, and the server request ID. To demonstrate the system, keep the API terminal visible beside the browser frontend; server logs contain only request/operation metadata and never book text, tokens, glossary text, or filenames.

## Required API

- `GET /` — concise JSON introduction, model metadata, and required endpoint purposes
- `GET /health` — API and llama.cpp/model status; returns `503` when the real model server is unavailable
- `POST /predict` — direct text prediction with `text`, `source_language`, `target_language`, optional `style_guide`, and optional glossary entries

Example direct prediction:

```powershell
$body = '{"text":"Alice enters the old garden.","source_language":"English","target_language":"Japanese","style_guide":"Preserve names and paragraph structure.","glossary":[{"source_term":"Alice","target_term":"アリス"}]}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 8
```

Representative response:

```json
{
  "translated_text": "アリス enters the old garden.",
  "paragraphs": ["アリス enters the old garden."],
  "source_language": "English",
  "target_language": "Japanese",
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "glossary_used": [{"source_term":"Alice","target_term":"アリス","category":"general","aliases":[]}]
}
```

## Document Workflow

- `POST /documents` — multipart upload using field `file` for TXT, MD, text PDF, or DRM-free EPUB
- `GET /documents/{document_id}/markdown` — canonical Markdown
- `GET /documents/{document_id}/chapters` — ordered chapters
- `POST /documents/{document_id}/chapters/{chapter_id}/analyze` — terminology analysis
- `GET /documents/{document_id}/glossary` — persistent glossary
- `PUT /documents/{document_id}/glossary/{source_term}` — edit glossary entry
- `POST /documents/{document_id}/chapters/{chapter_id}/translate` — saved chapter translation with language/style body
- `GET /documents/{document_id}/chapters/{chapter_id}/translation` — saved translation

Imports automatically split extracted sections larger than `MAX_MODEL_INPUT_CHARS` (default `30000`) at paragraph or sentence boundaries. The response includes warnings and ordered chapter parts such as `Section 1 (Part 1)`. Re-import the original source after configuring this limit to obtain the split representation; existing stored documents are not mutated.

The browser client at `http://127.0.0.1:3000/` provides upload, chapter analysis, glossary editing, and saved chapter translation. `api_demo.py` is a requests-based demonstration of all required endpoints and the optional document workflow:

```powershell
python .\api_demo.py .\book.md
```

## Tests

```powershell
python -m pytest -q
python -m compileall -q app tests api_demo.py
```

## Limitations

- llama.cpp must already be running with the configured model.
- PDF import has no OCR, DRM support, or layout preservation.
- Chapter and output sizes are limited by model/server context and token settings.
- The prototype has no production authentication or multi-user isolation.

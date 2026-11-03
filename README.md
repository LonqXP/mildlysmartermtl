# MTL-Proposal

## Chapter-by-Chapter Literary Translation Prototype

This is a local FastAPI prototype for importing a book, reviewing terminology chapter by chapter, and translating chapters with one configurable instruction-tuned model. Analysis and translation use the same backend. The default `mock` backend is deterministic and never downloads a model; `huggingface` is the assignment mode.

## Architecture

- `app/importers.py` extracts UTF-8 TXT/Markdown, text-layer PDF, and DRM-free EPUB spine content.
- `app/main.py` exposes the API and validates model output and glossary terms.
- `app/storage.py` persists documents, chapters, glossary entries, and translations in SQLite.
- `app/llm.py` contains the lazy Transformers backend and test/development mock backend.
- Runtime data is written to `.runtime/` and is ignored by git.

## Canonical Markdown

Every import is stored as the source of truth in this schema (`chapter-translation-v1`): YAML-like front matter containing `document_id`, `title`, and `schema`, followed by a document H1. Each ordered chapter is marked by `<!-- chapter_id: <stable id> -->`, followed by `## <ordinal>. <title>` and the original paragraphs. IDs are deterministic hashes of the source and chapter position/title.

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Optional assignment mode (downloads the selected model on first use):
pip install -r requirements-hf.txt
uvicorn app.main:app --reload
```

Environment variables: `LLM_BACKEND=mock|huggingface` (default `mock`), `HF_MODEL_ID` (default `Qwen/Qwen2.5-0.5B-Instruct`), `HF_DEVICE=auto`, `HF_DTYPE=auto|float16|bfloat16|float32`, `APP_DATA_DIR=.runtime`, `MAX_UPLOAD_BYTES=10485760`, and `MAX_PDF_PAGES=100`. Install `requirements-hf.txt` to use the real backend. The default is a small multilingual-capable instruction model and may be replaced with another instruction-tuned model appropriate to the available hardware. Model loading is lazy and occurs only on the first analysis or translation request.

## Endpoints

Required endpoints are `GET /`, `GET /health`, and `POST /predict`. Import is `POST /documents` with multipart field `file`. Other endpoints are `GET /documents/{id}/markdown`, `GET /documents/{id}/chapters`, `GET/PUT /documents/{id}/glossary[/{source_term}]`, and `POST /documents/{id}/chapters/{chapter_id}/analyze`.

Example:

```bash
curl -F "file=@book.md" http://127.0.0.1:8000/documents
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"document_id":"...","chapter_id":"..."}'
```

`/predict` returns translated canonical chapter Markdown, approved glossary entries used, and unresolved suggested/needs-review terms that occur in the chapter. Model prompts delimit document text as data. Uploads are capped; PDFs have a page cap; EPUB member count, member size, total uncompressed size, encryption, and paths are checked. FastAPI rejects malformed JSON with a clear 422 response.

## Tests and limitations

Run `pytest -q`. `api_demo.py` is a standalone Python `requests` API demo. The prototype does not perform OCR, preserve PDF layout, decrypt DRM, or provide production authentication/quotas. EPUB parsing intentionally supports common OPF/spine files rather than every EPUB variant. The mock backend demonstrates plumbing and glossary reuse but is not a translation model.
# mildlysmartermtl

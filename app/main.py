from datetime import datetime, timezone
import hashlib
import re
from typing import Any
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import load_settings
from .importers import extract
from .llm import GemmaBackend, MockBackend
from .storage import Store
from .terms import contains_any, normalize_term

settings = load_settings()
store = Store(settings.data_dir)
app = FastAPI(title="MTL-Proposal", description="Chapter-by-chapter literary translation prototype", version="0.1.0")
UI_FILE = Path(__file__).parent / "static" / "index.html"


def backend():
    if settings.backend.lower() == "mock": return MockBackend()
    if settings.backend in {"huggingface", "hf"}: return GemmaBackend(settings.remote_target, settings.hf_token, settings.timeout_seconds, settings.max_new_tokens)
    raise HTTPException(500, "LLM_BACKEND must be mock or huggingface")


def ident(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def canonical(doc_id: str, title: str, chapters: list[dict[str, Any]]) -> str:
    lines = ["---", f"document_id: {doc_id}", f"title: {title}", "schema: chapter-translation-v1", "---", "", f"# {title}", ""]
    for chapter in chapters:
        lines += [f"<!-- chapter_id: {chapter['id']} -->", f"## {chapter['ordinal']}. {chapter['title']}", "", chapter["text"], ""]
    return "\n".join(lines).strip() + "\n"


class GlossaryUpdate(BaseModel):
    source_term: str = Field(min_length=1)
    target_term: str = ""
    category: str = "general"
    aliases: list[str] = []
    status: str = "suggested"
    context: str = ""
    first_seen_chapter: str = ""
    notes: str = ""


class PredictRequest(BaseModel):
    document_id: str
    chapter_id: str
    style_guide: str = "Preserve meaning, paragraph breaks, names, and a consistent literary tone."
    source_language: str = "auto"
    target_language: str = "English"


class AnalyzeRequest(BaseModel):
    source_language: str = "auto"
    target_language: str = "English"
    style_guide: str = "Literary localization with consistent terminology."


@app.get("/")
def root():
    return {"name": app.title, "version": app.version, "docs": "/docs", "ui": "/ui", "backend": settings.backend, "llm_backend": "gemma-chat" if settings.backend in {"huggingface", "hf"} else "mock", "model_id": settings.model_id if settings.backend in {"huggingface", "hf"} else None, "endpoint_configured": bool(settings.endpoint_url)}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return UI_FILE.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "backend": settings.backend, "llm_backend": "gemma-chat" if settings.backend in {"huggingface", "hf"} else "mock", "model_id": settings.model_id if settings.backend in {"huggingface", "hf"} else None, "endpoint_configured": bool(settings.endpoint_url)}


@app.post("/documents", status_code=201)
async def import_document(file: UploadFile = File(...)):
    if not file.filename: raise HTTPException(400, "filename is required")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes: raise HTTPException(413, "upload exceeds size limit")
    try: title, sections = extract(file.filename, data, settings.max_pdf_pages)
    except Exception as exc: raise HTTPException(400, f"could not extract document: {exc}") from exc
    if not any(section.text.strip() for section in sections): raise HTTPException(400, "document contains no extractable text")
    doc_id = ident(file.filename, data.hex())
    chapters = [{"id": ident(doc_id, str(i), s.title), "document_id": doc_id, "ordinal": i, "title": s.title, "text": s.text} for i, s in enumerate(sections, 1) if s.text.strip()]
    doc = {"id": doc_id, "title": title, "filename": file.filename, "canonical": canonical(doc_id, title, chapters), "created_at": datetime.now(timezone.utc).isoformat()}
    try: store.add_document(doc, chapters)
    except Exception as exc: raise HTTPException(409, "document already exists or cannot be stored") from exc
    return {"document_id": doc_id, "title": title, "chapters": [{"id": x["id"], "ordinal": x["ordinal"], "title": x["title"]} for x in chapters]}


@app.get("/documents/{document_id}/markdown")
def get_markdown(document_id: str):
    doc = store.document(document_id)
    if not doc: raise HTTPException(404, "document not found")
    return {"document_id": document_id, "markdown": doc["canonical"]}


@app.get("/documents/{document_id}/chapters")
def list_chapters(document_id: str):
    if not store.document(document_id): raise HTTPException(404, "document not found")
    return {"chapters": [{"id": x["id"], "ordinal": x["ordinal"], "title": x["title"]} for x in store.chapters(document_id)]}


@app.get("/documents/{document_id}/glossary")
def get_glossary(document_id: str):
    if not store.document(document_id): raise HTTPException(404, "document not found")
    return {"entries": store.glossary(document_id)}


@app.put("/documents/{document_id}/glossary/{source_term}")
def update_glossary(document_id: str, source_term: str, item: GlossaryUpdate):
    if not store.document(document_id): raise HTTPException(404, "document not found")
    if source_term != item.source_term: raise HTTPException(400, "path and source_term must match")
    if item.status not in {"suggested", "approved", "rejected", "needs_review"}: raise HTTPException(400, "invalid glossary status")
    store.upsert_glossary(document_id, item.model_dump())
    return {"entry": next(x for x in store.glossary(document_id) if x["source_term"] == source_term)}


@app.post("/documents/{document_id}/chapters/{chapter_id}/analyze")
def analyze(document_id: str, chapter_id: str, request: AnalyzeRequest | None = None):
    chapter = store.chapter(document_id, chapter_id)
    if not chapter: raise HTTPException(404, "chapter not found")
    context = request or AnalyzeRequest()
    try: proposals = backend().analyze(chapter["text"], context.source_language, context.target_language, context.style_guide)
    except Exception as exc: raise HTTPException(502, f"invalid model analysis: {exc}") from exc
    existing = store.glossary(document_id)
    reused_matches, new_proposals, conflicts = [], [], []
    for proposal in proposals:
        if not isinstance(proposal, dict): continue
        source = str(proposal.get("source_term", "")).strip()
        raw_aliases = proposal.get("aliases", [])
        if not isinstance(raw_aliases, list): continue
        aliases = [str(x).strip() for x in raw_aliases if str(x).strip()]
        if not source or not contains_any(chapter["text"], [source, *aliases]): continue
        proposal_terms = {normalize_term(x) for x in [source, *aliases]}
        match = next((x for x in existing if proposal_terms.intersection(normalize_term(y) for y in [x["source_term"], *x["aliases"]])), None)
        if match:
            record = {"source_term": source, "matched_entry": match}
            (reused_matches if match["status"] == "approved" else conflicts).append(record)
            continue
        item = {"source_term": source, "target_term": str(proposal.get("target_term", source)), "category": str(proposal.get("category", "general")), "aliases": proposal.get("aliases", []), "status": "suggested", "context": str(proposal.get("context", chapter["text"][:160])), "first_seen_chapter": chapter_id, "notes": "Added by chapter analysis"}
        store.upsert_glossary(document_id, item); existing.append(item); new_proposals.append(item)
    return {"chapter_id": chapter_id, "reused_matches": reused_matches, "new_proposals": new_proposals, "needs_review": conflicts, "glossary": store.glossary(document_id)}


@app.post("/predict")
def predict(request: PredictRequest):
    chapter = store.chapter(request.document_id, request.chapter_id)
    if not chapter: raise HTTPException(404, "document or chapter not found")
    entries = store.glossary(request.document_id)
    glossary = [x for x in entries if x["status"] == "approved" and contains_any(chapter["text"], [x["source_term"], *x["aliases"]])]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chapter["text"].strip()) if p.strip()]
    try: translated = backend().translate(paragraphs, glossary, request.style_guide, request.source_language, request.target_language)
    except Exception as exc: raise HTTPException(502, f"invalid model translation: {exc}") from exc
    if len(translated) != len(paragraphs) or not all(isinstance(x, str) for x in translated): raise HTTPException(502, "model changed paragraph structure")
    warnings = [f"unresolved glossary term: {x['source_term']}" for x in entries if x["status"] in {"suggested", "needs_review"} and contains_any(chapter["text"], [x["source_term"], *x["aliases"]])]
    markdown = f"## {chapter['ordinal']}. {chapter['title']}\n\n" + "\n\n".join(translated) + "\n"
    store.save_translation(request.document_id, request.chapter_id, markdown, warnings)
    return {"document_id": request.document_id, "chapter_id": request.chapter_id, "markdown": markdown, "warnings": warnings, "glossary_used": glossary}


@app.get("/documents/{document_id}/chapters/{chapter_id}/translation")
def get_translation(document_id: str, chapter_id: str):
    if not store.chapter(document_id, chapter_id): raise HTTPException(404, "chapter not found")
    result = store.translation(document_id, chapter_id)
    if not result: raise HTTPException(404, "translation not found")
    return result

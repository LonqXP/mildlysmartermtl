from datetime import datetime, timezone
import hashlib
import logging
import re
import time
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_settings
from .importers import extract, split_oversized_sections
from .llm import LlamaCppBackend, MockBackend
from .storage import Store
from .terms import contains_any, normalize_term

settings = load_settings()
store = Store(settings.data_dir)
app = FastAPI(title="mildlysmartermtl", description="mildlysmartermtl chapter-by-chapter literary translation JSON API", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.frontend_origins), allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
_backend_instance = None
logger = logging.getLogger("uvicorn.error.mildlysmartermtl")
CANONICAL_SCHEMA = "chapter-translation-v2"


@app.middleware("http")
async def request_observability(request, call_next):
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["Server-Timing"] = f"app;dur={duration_ms:.1f}"
    logger.info("request method=%s path=%s status=%d duration_ms=%.1f request_id=%s", request.method, request.url.path, response.status_code, duration_ms, request_id)
    return response


def backend():
    global _backend_instance
    if _backend_instance is None:
        if settings.backend == "mock": _backend_instance = MockBackend()
        elif settings.backend == "llamacpp": _backend_instance = LlamaCppBackend(settings.base_url, settings.model, settings.api_key, settings.connect_timeout_seconds, settings.read_timeout_seconds, settings.analysis_max_tokens, settings.translation_max_tokens, settings.translation_batch_chars, settings.disable_thinking, settings.json_mode)
        else: raise HTTPException(500, "LLM_BACKEND must be mock or llamacpp")
    return _backend_instance


def warmup_backend():
    current = backend()
    return current.warmup() if hasattr(current, "warmup") else {"status": "ready", "backend": settings.backend}


def ident(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def canonical(doc_id: str, title: str, chapters: list[dict[str, Any]]) -> str:
    lines = ["---", f"document_id: {doc_id}", f"title: {title}", f"schema: {CANONICAL_SCHEMA}", "---", "", f"# {title}", ""]
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


class GlossaryEntry(BaseModel):
    source_term: str = Field(min_length=1)
    target_term: str = Field(min_length=1)
    category: str = "general"
    aliases: list[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    style_guide: str = "Preserve meaning, paragraph breaks, names, and a consistent literary tone."
    source_language: str = "auto"
    target_language: str = "English"
    glossary: list[GlossaryEntry] = Field(default_factory=list)


class ChapterTranslateRequest(BaseModel):
    source_language: str = "auto"
    target_language: str = "English"
    style_guide: str = "Preserve meaning, paragraph breaks, names, and a consistent literary tone."


class AnalyzeRequest(BaseModel):
    source_language: str = "auto"
    target_language: str = "English"
    style_guide: str = "Literary localization with consistent terminology."


@app.get("/")
def root():
    return {"name": app.title, "description": "JSON API for chapter import, terminology analysis, and glossary-guided literary translation.", "backend": settings.backend, "model": settings.model if settings.backend == "llamacpp" else "mock", "model_source": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF" if settings.backend == "llamacpp" else None, "required_endpoints": {"GET /": "API introduction", "GET /health": "API and model health", "POST /predict": "direct text prediction"}, "frontend": "http://127.0.0.1:3000/"}


@app.get("/health")
def health():
    result = backend().health()
    if settings.backend == "llamacpp" and result.get("status") != "ok": raise HTTPException(503, detail=result)
    return result


@app.post("/documents", status_code=201)
async def import_document(file: UploadFile = File(...), response: Response = None):
    if not file.filename: raise HTTPException(400, "filename is required")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes: raise HTTPException(413, "upload exceeds size limit")
    try: title, sections = extract(file.filename, data, settings.max_pdf_pages)
    except Exception as exc: raise HTTPException(400, f"could not extract document: {exc}") from exc
    doc_id = ident(CANONICAL_SCHEMA, file.filename, data.hex())
    existing = store.document(doc_id)
    if existing:
        if response is not None: response.status_code = 200
        return {"document_id": doc_id, "title": existing["title"], "chapters": [{"id": x["id"], "ordinal": x["ordinal"], "title": x["title"]} for x in store.chapters(doc_id)], "already_exists": True, "warnings": ["This exact document was already imported; using the existing normalized copy."]}
    sections, warnings = split_oversized_sections(sections, settings.max_model_input_chars)
    if not any(section.text.strip() for section in sections): raise HTTPException(400, "document contains no extractable text")
    chapters = [{"id": ident(doc_id, str(i), s.title), "document_id": doc_id, "ordinal": i, "title": s.title, "text": s.text} for i, s in enumerate(sections, 1) if s.text.strip()]
    doc = {"id": doc_id, "title": title, "filename": file.filename, "canonical": canonical(doc_id, title, chapters), "created_at": datetime.now(timezone.utc).isoformat()}
    try: store.add_document(doc, chapters)
    except Exception as exc:
        if store.document(doc_id):
            if response is not None: response.status_code = 200
            return {"document_id": doc_id, "title": doc["title"], "chapters": [{"id": x["id"], "ordinal": x["ordinal"], "title": x["title"]} for x in store.chapters(doc_id)], "already_exists": True, "warnings": ["This exact document was already imported; using the existing normalized copy."]}
        logger.error("document storage failed document_id=%s chapter_count=%d error_type=%s", doc_id, len(chapters), type(exc).__name__)
        raise HTTPException(500, "document could not be stored") from exc
    return {"document_id": doc_id, "title": title, "chapters": [{"id": x["id"], "ordinal": x["ordinal"], "title": x["title"]} for x in chapters], "already_exists": False, "warnings": warnings}


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
    ensure_model_input(chapter["text"])
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
    ensure_model_input(request.text)
    paragraphs = split_paragraphs(request.text)
    try: translated = backend().translate(paragraphs, [entry.model_dump() for entry in request.glossary], request.style_guide, request.source_language, request.target_language)
    except Exception as exc: raise HTTPException(502, f"invalid model translation: {exc}") from exc
    if len(translated) != len(paragraphs) or not all(isinstance(x, str) for x in translated): raise HTTPException(502, "model changed paragraph structure")
    return {"translated_text": "\n\n".join(translated), "paragraphs": translated, "source_language": request.source_language, "target_language": request.target_language, "model": settings.model if settings.backend == "llamacpp" else "mock", "glossary_used": [entry.model_dump() for entry in request.glossary]}


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    if not paragraphs: raise HTTPException(422, "text must contain non-whitespace content")
    return paragraphs


def ensure_model_input(text: str):
    actual = len(text)
    if actual > settings.max_model_input_chars:
        raise HTTPException(413, f"model input is {actual} characters; limit is {settings.max_model_input_chars}. Split or re-import the source into smaller sections.")


@app.post("/documents/{document_id}/chapters/{chapter_id}/translate")
def translate_chapter(document_id: str, chapter_id: str, request: ChapterTranslateRequest):
    chapter = store.chapter(document_id, chapter_id)
    if not chapter: raise HTTPException(404, "document or chapter not found")
    ensure_model_input(chapter["text"])
    entries = store.glossary(document_id)
    glossary = [entry for entry in entries if entry["status"] == "approved" and contains_any(chapter["text"], [entry["source_term"], *entry["aliases"]])]
    try: translated = backend().translate(split_paragraphs(chapter["text"]), glossary, request.style_guide, request.source_language, request.target_language)
    except Exception as exc: raise HTTPException(502, f"invalid model translation: {exc}") from exc
    paragraphs = split_paragraphs(chapter["text"])
    if len(translated) != len(paragraphs) or not all(isinstance(x, str) for x in translated): raise HTTPException(502, "model changed paragraph structure")
    warnings = [f"unresolved glossary term: {entry['source_term']}" for entry in entries if entry["status"] in {"suggested", "needs_review"} and contains_any(chapter["text"], [entry["source_term"], *entry["aliases"]])]
    markdown = f"## {chapter['ordinal']}. {chapter['title']}\n\n" + "\n\n".join(translated) + "\n"
    store.save_translation(document_id, chapter_id, markdown, warnings)
    return {"document_id": document_id, "chapter_id": chapter_id, "markdown": markdown, "warnings": warnings, "glossary_used": glossary, "source_language": request.source_language, "target_language": request.target_language}


@app.get("/documents/{document_id}/chapters/{chapter_id}/translation")
def get_translation(document_id: str, chapter_id: str):
    if not store.chapter(document_id, chapter_id): raise HTTPException(404, "chapter not found")
    result = store.translation(document_id, chapter_id)
    if not result: raise HTTPException(404, "translation not found")
    return result

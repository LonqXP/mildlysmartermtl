import io
import os
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ["APP_DATA_DIR"] = ".test-runtime"
os.environ["LLM_BACKEND"] = "mock"
shutil.rmtree(".test-runtime", ignore_errors=True)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def upload(name, text):
    response = client.post("/documents", files={"file": (name, io.BytesIO(text.encode()), "text/plain")})
    assert response.status_code == 201, response.text
    return response.json()


def test_health_and_txt_fallback():
    assert client.get("/health").json()["status"] == "ok"
    doc = upload("book.txt", "A quiet opening paragraph.\n\nA second paragraph.")
    assert len(doc["chapters"]) == 1
    markdown = client.get(f"/documents/{doc['document_id']}/markdown").json()["markdown"]
    assert "schema: chapter-translation-v2" in markdown
    assert "A quiet opening" in markdown


def test_duplicate_upload_is_idempotent_and_reuses_chapters():
    source = b"Duplicate-safe paragraph.\n\nSecond paragraph."
    first = client.post("/documents", files={"file": ("duplicate.txt", source, "text/plain")})
    second = client.post("/documents", files={"file": ("duplicate.txt", source, "text/plain")})
    assert first.status_code == 201
    assert second.status_code == 200
    first_data, second_data = first.json(), second.json()
    assert first_data["already_exists"] is False
    assert second_data["already_exists"] is True
    assert first_data["document_id"] == second_data["document_id"]
    assert len(first_data["chapters"]) == len(second_data["chapters"])
    assert len(client.get(f"/documents/{first_data['document_id']}/chapters").json()["chapters"]) == len(first_data["chapters"])
    assert "already imported" in second_data["warnings"][0]


def test_root_introduction_and_frontend_file():
    response = client.get("/")
    data = response.json()
    assert data["required_endpoints"]["POST /predict"] == "direct text prediction"
    frontend = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "http://127.0.0.1:8000" in frontend
    assert 'id="upload-form"' in frontend
    assert 'id="analyze-button"' in frontend
    assert 'id="translate-button"' in frontend
    assert "already_exists" in frontend
    assert "needs-review" in frontend
    assert "Review required" in frontend
    assert "review-instruction" in frontend
    assert 'id="glossary-review-count"' in frontend
    assert "needs_review: 'Needs review'" in frontend
    assert '<option value="approved">Approved</option>' in frontend
    assert '<option value="rejected">Rejected</option>' in frontend
    assert client.get("/").headers["x-request-id"]
    assert "app;dur=" in client.get("/").headers["server-timing"]


def test_removed_documentation_and_ui_routes_are_404():
    for path in ("/ui", "/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_cors_allows_configured_frontend_origin():
    response = client.get("/", headers={"Origin": "http://127.0.0.1:3000"})
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


def test_health_unavailable_returns_503(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "settings", SimpleNamespace(backend="llamacpp"))
    monkeypatch.setattr(main, "backend", lambda: SimpleNamespace(health=lambda: {"status": "offline", "backend": "llamacpp"}))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "offline"


def test_markdown_chapters_and_glossary_reuse():
    doc = upload("book.md", "# Chapter One\n\nAlice meets Bob.\n\n## Chapter Two\n\nBob returns.")
    assert len(doc["chapters"]) == 2
    chapter = doc["chapters"][0]
    analysis = client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/analyze", json={"source_language": "Japanese", "target_language": "English"})
    assert analysis.status_code == 200
    update = client.put(f"/documents/{doc['document_id']}/glossary/Alice", json={"source_term": "Alice", "target_term": "Alicia", "status": "approved"})
    assert update.status_code == 200
    prediction = client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/translate", json={"source_language": "Japanese", "target_language": "English"})
    assert prediction.status_code == 200
    assert "Alicia" in prediction.json()["markdown"]


def test_direct_predict_validates_and_invokes_mock_backend():
    response = client.post("/predict", json={"text": "Alice meets Bob.\n\nBob returns.", "source_language": "English", "target_language": "Japanese", "glossary": [{"source_term": "Alice", "target_term": "Alicia"}]})
    assert response.status_code == 200
    data = response.json()
    assert data["paragraphs"] == ["Alicia meets Bob.", "Bob returns."]
    assert data["source_language"] == "English"
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_oversized_model_inputs_return_413(monkeypatch):
    import app.main as main
    doc = upload("limit.txt", "A paragraph that is intentionally longer than the test limit.")
    monkeypatch.setattr(main, "settings", replace(main.settings, max_model_input_chars=10))
    assert client.post("/predict", json={"text": "01234567890"}).status_code == 413
    chapter = doc["chapters"][0]
    assert client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/analyze").status_code == 413
    assert client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/translate", json={}).status_code == 413


def test_large_import_is_split_in_order_without_text_loss(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "settings", replace(main.settings, max_model_input_chars=100))
    source = "\n\n".join(f"Paragraph {index:03d} with stable source text." for index in range(12))
    response = client.post("/documents", files={"file": ("large.txt", source.encode(), "text/plain")})
    assert response.status_code == 201
    data = response.json()
    assert data["warnings"] and len(data["chapters"]) > 1
    assert data["chapters"][0]["title"] == "Section 1 (Part 1)"
    assert data["chapters"][1]["title"] == "Section 1 (Part 2)"
    stored = client.get(f"/documents/{data['document_id']}/chapters").json()["chapters"]
    assert [item["ordinal"] for item in stored] == list(range(1, len(stored) + 1))
    old_id = main.ident("large.txt", source.encode().hex())
    assert data["document_id"] != old_id


def test_rejects_unsupported_and_empty():
    assert client.post("/documents", files={"file": ("x.docx", b"text", "application/octet-stream")}).status_code == 400
    assert client.post("/documents", files={"file": ("x.txt", b"", "text/plain")}).status_code == 400


def test_cjk_alias_replacement_and_only_relevant_unresolved_warning():
    doc = upload("cjk.txt", "日本語と 한국어 are discussed here.")
    chapter = doc["chapters"][0]
    base = f"/documents/{doc['document_id']}/glossary"
    assert client.put(base + "/日本語", json={"source_term": "日本語", "target_term": "Japanese", "aliases": ["일본어"], "status": "approved"}).status_code == 200
    assert client.put(base + "/한국어", json={"source_term": "한국어", "target_term": "Korean", "status": "suggested"}).status_code == 200
    assert client.put(base + "/unseen", json={"source_term": "unseen", "target_term": "unused", "status": "approved"}).status_code == 200
    result = client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/translate", json={}).json()
    assert "Japanese" in result["markdown"]
    assert result["warnings"] == ["unresolved glossary term: 한국어"]


def test_analyze_reports_approved_alias_reuse(monkeypatch):
    import app.main as main

    doc = upload("reuse.txt", "The term 일语 appears here.")
    chapter = doc["chapters"][0]
    client.put(f"/documents/{doc['document_id']}/glossary/日本語", json={"source_term": "日本語", "target_term": "Japanese", "aliases": ["일语"], "status": "approved"})

    class FakeBackend:
        def analyze(self, text, source_language="auto", target_language="English", style_guide=""):
            return [{"source_term": "일语", "target_term": "Japanese", "aliases": [], "category": "term", "context": text}]

    monkeypatch.setattr(main, "backend", lambda: FakeBackend())
    result = client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/analyze").json()
    assert result["new_proposals"] == []
    assert result["reused_matches"][0]["matched_entry"]["source_term"] == "日本語"


def test_synthetic_epub_spine_and_unsafe_path():
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>')
        archive.writestr("OPS/book.opf", '<package><manifest><item href="chap2.xhtml" id="b"/><item id="a" href="chap1.xhtml"/></manifest><spine><itemref idref="a"/><itemref idref="b"/></spine></package>')
        archive.writestr("OPS/chap1.xhtml", "<html><body><h1>First</h1><p>日本語.</p></body></html>")
        archive.writestr("OPS/chap2.xhtml", "<html><body><p>Second.</p></body></html>")
    response = client.post("/documents", files={"file": ("book.epub", data.getvalue(), "application/epub+zip")})
    assert response.status_code == 201
    assert [x["title"] for x in response.json()["chapters"]] == ["Section 1", "Section 1"]
    unsafe = io.BytesIO()
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="../book.opf"/></rootfiles></container>')
    assert client.post("/documents", files={"file": ("unsafe.epub", unsafe.getvalue(), "application/epub+zip")}).status_code == 400

import io
import os
import shutil
import zipfile

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
    assert "schema: chapter-translation-v1" in markdown
    assert "A quiet opening" in markdown


def test_ui_route_returns_branded_html_controls():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "MTL-Proposal" in response.text
    assert 'id="upload-form"' in response.text
    assert 'id="analyze-button"' in response.text
    assert 'id="translate-button"' in response.text


def test_markdown_chapters_and_glossary_reuse():
    doc = upload("book.md", "# Chapter One\n\nAlice meets Bob.\n\n## Chapter Two\n\nBob returns.")
    assert len(doc["chapters"]) == 2
    chapter = doc["chapters"][0]
    analysis = client.post(f"/documents/{doc['document_id']}/chapters/{chapter['id']}/analyze", json={"source_language": "Japanese", "target_language": "English"})
    assert analysis.status_code == 200
    update = client.put(f"/documents/{doc['document_id']}/glossary/Alice", json={"source_term": "Alice", "target_term": "Alicia", "status": "approved"})
    assert update.status_code == 200
    prediction = client.post("/predict", json={"document_id": doc["document_id"], "chapter_id": chapter["id"], "source_language": "Japanese", "target_language": "English"})
    assert prediction.status_code == 200
    assert "Alicia" in prediction.json()["markdown"]


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
    result = client.post("/predict", json={"document_id": doc["document_id"], "chapter_id": chapter["id"]}).json()
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

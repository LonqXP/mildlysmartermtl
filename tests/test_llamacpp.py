import json
import httpx
import pytest

from app.llm import LlamaCppBackend


def backend_with(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234")
    return LlamaCppBackend("http://127.0.0.1:1234", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", analysis_max_tokens=2048, translation_max_tokens=64, client=client)


def test_request_shape_prompt_and_json_normalization():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "```json\n{\"terms\": [{\"source_term\": \"日本語\", \"target_term\": \"Japanese\", \"category\": \"term\", \"aliases\": [], \"context\": \"x\"}]}\n```"}}]})

    result = backend_with(handler).analyze("日本語", "Japanese", "English", "Keep literary names")
    payload = json.loads(requests[0].content)
    assert result[0]["source_term"] == "日本語"
    assert payload["model"] == "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 2048
    assert "top 15-20" in payload["messages"][0]["content"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == {"type": "json_object"}
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert "Source language: Japanese" in payload["messages"][1]["content"]
    assert "APPROVED GLOSSARY" not in payload["messages"][1]["content"]


def test_translation_uses_disabled_thinking_and_glossary_prompt():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"translations":[{"id":"p0f0","text":"Translated"}]}'}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", analysis_max_tokens=2048, translation_max_tokens=10, disable_thinking=True, client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    assert backend.translate(["원문"], [{"source_term": "원문", "target_term": "Original", "aliases": []}], "Formal", "Korean", "English") == ["Translated"]
    payload = json.loads(requests[0].content)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Korean" in payload["messages"][1]["content"]
    assert "Original" in payload["messages"][1]["content"]


def test_health_discovers_loaded_model():
    def handler(request):
        if request.url.path == "/health": return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"data": [{"id": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"}]})

    result = backend_with(handler).health()
    assert result["status"] == "ok"
    assert result["loaded_models"] == ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"]


def test_timeout_maps_to_clear_error():
    def handler(request): raise httpx.ReadTimeout("read timeout", request=request)
    backend = backend_with(handler)
    with pytest.raises(RuntimeError, match="timed out"):
        backend.analyze("safe test")


def test_analysis_length_finish_reason_is_explicit():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{\"terms\":["}}], "usage": {"completion_tokens": 2048}})
    with pytest.raises(RuntimeError, match="analysis output was truncated at 2048 tokens"):
        backend_with(handler).analyze("safe test")


def test_analysis_recovers_only_complete_term_objects():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"terms":[{"source_term":"Alice","target_term":"Alicia","category":"name","aliases":[],"context":"Alice"},{"source_term":"Broken"'}}]})
    terms = backend_with(handler).analyze("Alice")
    assert [term["source_term"] for term in terms] == ["Alice"]


def test_translation_batches_and_recombines_long_paragraphs():
    calls = []

    def handler(request):
        payload = json.loads(request.content); calls.append(payload)
        raw = payload["messages"][1]["content"].split("TRANSLATION INPUT JSON START\n", 1)[1].split("\nTRANSLATION INPUT JSON END", 1)[0]
        items = json.loads(raw)
        translations = [{"id": item["id"], "text": f"T{item['id']}"} for item in reversed(items)]
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"translations": translations})}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", analysis_max_tokens=2048, translation_max_tokens=8192, translation_batch_chars=20, client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    result = backend.translate(["One sentence. Two sentence.", "Short."], [], "Style", "English", "Japanese")
    assert len(calls) > 1
    assert len(result) == 2
    assert result[0] == "Tp0f0 Tp0f1"
    assert result[1] == "Tp1f0"


def test_mismatched_many_item_output_bisects_and_restores_count():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        raw = payload["messages"][1]["content"].split("TRANSLATION INPUT JSON START\n", 1)[1].split("\nTRANSLATION INPUT JSON END", 1)[0]
        items = json.loads(raw); calls.append(len(items))
        if len(calls) == 1:
            content = json.dumps({"paragraphs": ["wrong"] * 58})
        else:
            content = json.dumps({"translations": [{"id": item["id"], "text": item["text"]} for item in items]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", translation_batch_chars=10000, client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    paragraphs = [f"Paragraph {index}." for index in range(15)]
    assert backend.translate(paragraphs, [], "Style") == paragraphs
    assert calls[0] == 15 and len(calls) > 1


def test_duplicate_or_missing_ids_fail_after_bounded_retry():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"translations":[{"id":"wrong","text":""}]}'}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    with pytest.raises(ValueError, match="no usable text fragments"):
        backend.translate(["One.", "Two."], [], "Style")


def test_single_item_fragment_texts_are_combined():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"translations":[{"id":"p0f0","text":"first"},{"id":"p0f0","text":"second"}]}'}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    assert backend.translate(["One."], [], "Style") == ["first second"]


def test_single_item_invented_fragment_ids_are_combined():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"translations":[{"id":"invented-a","text":"first"},{"id":"invented-b","text":"  "},{"id":"invented-c","text":"second"}]}'}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    assert backend.translate(["One."], [], "Style") == ["first second"]


def test_single_item_legacy_paragraph_fragments_are_combined():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"paragraphs":["first", "", " second "]}'}}]})

    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234"))
    assert backend.translate(["One."], [], "Style") == ["first second"]


def test_oversized_paragraph_without_sentence_boundary_is_rejected():
    backend = LlamaCppBackend("http://127.0.0.1:1234", "model", translation_batch_chars=10, client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))))
    with pytest.raises(ValueError, match="safe sentence boundary"):
        backend.translate(["x" * 11], [], "Style")


def test_sanitized_model_error_and_response_format_retry_detail():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1: return httpx.Response(400, json={"error": "response_format unsupported; prompt secret should not matter"})
        return httpx.Response(400, json={"error": "context window exceeded"})

    with pytest.raises(RuntimeError, match="HTTP 400.*context window exceeded") as error:
        backend_with(handler).analyze("safe test")
    assert "prompt secret" not in str(error.value)
    assert len(calls) == 2


def test_context_error_is_actionable():
    def handler(request): return httpx.Response(400, json={"error": "prompt is too long for context window"})
    with pytest.raises(RuntimeError, match="prompt as too large"):
        backend_with(handler).analyze("safe test")


def test_main_backend_is_cached(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "_backend_instance", None)
    first = main.backend()
    assert first is main.backend()

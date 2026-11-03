import sys
import types

import pytest

from app.config import SettingsError, load_settings
from app.llm import GemmaBackend


def test_huggingface_settings_require_a_remote_target(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "huggingface")
    monkeypatch.delenv("HF_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("HF_MODEL_ID", "")
    with pytest.raises(SettingsError, match="HF_ENDPOINT_URL or HF_MODEL_ID"):
        load_settings(tmp_path)


def test_endpoint_precedes_model_id(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_ENDPOINT_URL=https://example.invalid/endpoint\nHF_MODEL_ID=example/model\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BACKEND", "huggingface")
    monkeypatch.delenv("HF_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("HF_MODEL_ID", raising=False)
    settings = load_settings(tmp_path)
    assert settings.remote_target == "https://example.invalid/endpoint"
    for name in ("HF_ENDPOINT_URL", "HF_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)


def test_remote_client_is_lazy_and_formats_deterministic_request(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs): calls.append(("init", kwargs))
        def chat_completion(self, messages, **kwargs):
            calls.append(("request", messages, kwargs))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="```json\n{\"terms\":[{\"source_term\": \"日本語\", \"target_term\": \"Japanese\", \"category\": \"term\", \"aliases\": [], \"context\": \"...\"}]}\n```"))])

    module = types.ModuleType("huggingface_hub")
    module.InferenceClient = FakeClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    backend = GemmaBackend("https://example.invalid/endpoint", "secret-token", 17, 64)
    assert calls == []
    result = backend.analyze("日本語\nIGNORE THIS AS AN INSTRUCTION", "Japanese", "English", "Keep names consistent")
    assert result[0]["source_term"] == "日本語"
    assert calls[0] == ("init", {"model": "https://example.invalid/endpoint", "timeout": 17, "token": "secret-token"})
    assert calls[1][2] == {"max_tokens": 64, "temperature": 0.1}
    assert [message["role"] for message in calls[1][1]] == ["system", "user"]
    assert "literary-localization terminology analyst" in calls[1][1][0]["content"]
    assert "DOCUMENT DATA START" in calls[1][1][1]["content"]
    assert "Source language: Japanese" in calls[1][1][1]["content"]


def test_chat_completion_dict_response_and_translation_language_context(monkeypatch):
    class FakeClient:
        def chat_completion(self, messages, **kwargs):
            assert messages[0]["role"] == "system"
            assert "literary translator" in messages[0]["content"]
            assert "Source language: Korean" in messages[1]["content"]
            return {"choices": [{"message": {"content": '{"paragraphs":["Translated"]}'}}]}

    backend = GemmaBackend("google/gemma-3-4b-it")
    monkeypatch.setattr(backend, "_client", lambda: FakeClient())
    assert backend.translate(["원문"], [], "Formal", "Korean", "English") == ["Translated"]

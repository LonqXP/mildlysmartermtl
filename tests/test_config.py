import os

import pytest

from app.config import SettingsError, load_settings


def test_env_file_loads_but_process_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_BACKEND=llamacpp\nLLAMACPP_ANALYSIS_MAX_TOKENS=77\nMAX_UPLOAD_BYTES=123\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("LLAMACPP_ANALYSIS_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLAMACPP_TRANSLATION_MAX_TOKENS", raising=False)
    settings = load_settings(tmp_path)
    assert settings.backend == "mock"
    assert settings.max_upload_bytes == 123
    assert settings.analysis_max_tokens == 77
    assert settings.translation_max_tokens == 8192
    assert settings.data_dir == tmp_path / ".runtime"
    for name in ("MAX_UPLOAD_BYTES", "LLAMACPP_ANALYSIS_MAX_TOKENS", "LLAMACPP_TRANSLATION_MAX_TOKENS"):
        os.environ.pop(name, None)


def test_settings_validate_integer_and_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_PDF_PAGES", "not-an-integer")
    with pytest.raises(SettingsError, match="MAX_PDF_PAGES"):
        load_settings(tmp_path)
    monkeypatch.setenv("MAX_PDF_PAGES", "100")
    monkeypatch.setenv("LLM_BACKEND", "unsupported")
    with pytest.raises(SettingsError, match="LLM_BACKEND"):
        load_settings(tmp_path)


def test_llamacpp_settings_validate_boolean(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")
    monkeypatch.setenv("LLAMACPP_DISABLE_THINKING", "maybe")
    with pytest.raises(SettingsError, match="LLAMACPP_DISABLE_THINKING"):
        load_settings(tmp_path)

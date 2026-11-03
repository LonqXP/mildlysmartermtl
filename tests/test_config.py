import os

import pytest

from app.config import SettingsError, load_settings


def test_env_file_loads_but_process_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LLM_BACKEND=huggingface\nMAX_UPLOAD_BYTES=123\nHF_MAX_NEW_TOKENS=77\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("HF_MAX_NEW_TOKENS", raising=False)
    settings = load_settings(tmp_path)
    assert settings.backend == "mock"
    assert settings.max_upload_bytes == 123
    assert settings.max_new_tokens == 77
    assert settings.data_dir == tmp_path / ".runtime"
    for name in ("MAX_UPLOAD_BYTES", "HF_MAX_NEW_TOKENS"):
        os.environ.pop(name, None)


def test_settings_validate_integer_and_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_PDF_PAGES", "not-an-integer")
    with pytest.raises(SettingsError, match="MAX_PDF_PAGES"):
        load_settings(tmp_path)
    monkeypatch.setenv("MAX_PDF_PAGES", "100")
    monkeypatch.setenv("LLM_BACKEND", "unsupported")
    with pytest.raises(SettingsError, match="LLM_BACKEND"):
        load_settings(tmp_path)

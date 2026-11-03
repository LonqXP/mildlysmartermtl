from dataclasses import dataclass
from pathlib import Path
import os
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SettingsError(ValueError):
    """Raised when an application setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    backend: str
    base_url: str
    model: str
    api_key: str | None
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_upload_bytes: int
    max_pdf_pages: int
    analysis_max_tokens: int
    translation_max_tokens: int
    translation_batch_chars: int
    disable_thinking: bool
    json_mode: bool
    frontend_origins: tuple[str, ...]
    max_model_input_chars: int


def _positive_number(name: str, default: str, integer: bool = False):
    raw = os.getenv(name, default).strip()
    try: value = int(raw) if integer else float(raw)
    except ValueError as exc: raise SettingsError(f"{name} must be a {'integer' if integer else 'number'}") from exc
    if value <= 0: raise SettingsError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw not in {"1", "true", "yes", "on", "0", "false", "no", "off"}: raise SettingsError(f"{name} must be a boolean")
    return raw in {"1", "true", "yes", "on"}


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or PROJECT_ROOT).resolve()
    load_dotenv(dotenv_path=root / ".env", override=False)
    backend = os.getenv("LLM_BACKEND", "mock").strip().lower()
    if backend not in {"mock", "llamacpp"}: raise SettingsError("LLM_BACKEND must be mock or llamacpp")
    base_url = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:1234").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc: raise SettingsError("LLAMACPP_BASE_URL must be an http(s) URL")
    model = os.getenv("LLAMACPP_MODEL", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf").strip()
    if not model: raise SettingsError("LLAMACPP_MODEL must not be empty")
    origins = tuple(item.strip() for item in os.getenv("FRONTEND_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000").split(",") if item.strip())
    if not origins or "*" in origins: raise SettingsError("FRONTEND_ORIGINS must contain explicit origins")
    for origin in origins:
        parsed_origin = urlparse(origin)
        if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.netloc or parsed_origin.path not in {"", "/"}:
            raise SettingsError("FRONTEND_ORIGINS must contain valid explicit origins")
    data_dir = Path(os.getenv("APP_DATA_DIR", ".runtime")).expanduser()
    if not data_dir.is_absolute(): data_dir = root / data_dir
    return Settings(
        project_root=root, data_dir=data_dir, backend=backend, base_url=base_url, model=model,
        api_key=os.getenv("LLAMACPP_API_KEY") or None,
        connect_timeout_seconds=_positive_number("LLAMACPP_CONNECT_TIMEOUT_SECONDS", "5"),
        read_timeout_seconds=_positive_number("LLAMACPP_READ_TIMEOUT_SECONDS", "600"),
        max_upload_bytes=_positive_number("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024), integer=True),
        max_pdf_pages=_positive_number("MAX_PDF_PAGES", "100", integer=True),
        analysis_max_tokens=_positive_number("LLAMACPP_ANALYSIS_MAX_TOKENS", os.getenv("LLAMACPP_MAX_TOKENS", "2048"), integer=True),
        translation_max_tokens=_positive_number("LLAMACPP_TRANSLATION_MAX_TOKENS", os.getenv("LLAMACPP_MAX_TOKENS", "8192"), integer=True),
        translation_batch_chars=_positive_number("LLAMACPP_TRANSLATION_BATCH_CHARS", "10000", integer=True),
        disable_thinking=_boolean("LLAMACPP_DISABLE_THINKING", True),
        json_mode=_boolean("LLAMACPP_JSON_MODE", True),
        frontend_origins=origins,
        max_model_input_chars=_positive_number("MAX_MODEL_INPUT_CHARS", "30000", integer=True),
    )

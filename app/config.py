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
    endpoint_url: str | None
    model_id: str | None
    hf_token: str | None
    timeout_seconds: float
    max_upload_bytes: int
    max_pdf_pages: int
    max_new_tokens: int

    @property
    def remote_target(self) -> str | None:
        """Dedicated endpoint wins over a hosted model ID when both are set."""
        return self.endpoint_url or self.model_id


def _positive_number(name: str, default: str, integer: bool = False):
    raw = os.getenv(name, default).strip()
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        kind = "integer" if integer else "number"
        raise SettingsError(f"{name} must be a {kind}") from exc
    if value <= 0: raise SettingsError(f"{name} must be greater than zero")
    return value


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or PROJECT_ROOT).resolve()
    load_dotenv(dotenv_path=root / ".env", override=False)
    backend = os.getenv("LLM_BACKEND", "mock").strip().lower()
    if backend not in {"mock", "huggingface", "hf"}:
        raise SettingsError("LLM_BACKEND must be mock or huggingface")
    endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip() or None
    if endpoint_url:
        parsed = urlparse(endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SettingsError("HF_ENDPOINT_URL must be an http(s) URL")
    model_id = os.getenv("HF_MODEL_ID", "google/gemma-3-4b-it").strip() or None
    if backend in {"huggingface", "hf"} and not endpoint_url and not model_id:
        raise SettingsError("huggingface backend requires HF_ENDPOINT_URL or HF_MODEL_ID")
    data_dir = Path(os.getenv("APP_DATA_DIR", ".runtime")).expanduser()
    if not data_dir.is_absolute(): data_dir = root / data_dir
    return Settings(
        project_root=root,
        data_dir=data_dir,
        backend=backend,
        endpoint_url=endpoint_url,
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN") or None,
        timeout_seconds=_positive_number("HF_TIMEOUT_SECONDS", "120"),
        max_upload_bytes=_positive_number("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024), integer=True),
        max_pdf_pages=_positive_number("MAX_PDF_PAGES", "100", integer=True),
        max_new_tokens=_positive_number("HF_MAX_NEW_TOKENS", "512", integer=True),
    )

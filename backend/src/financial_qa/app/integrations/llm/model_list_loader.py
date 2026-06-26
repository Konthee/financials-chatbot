"""Load ``assets/model_list.yaml`` and expand ``${ENV}`` placeholders for LiteLLM."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_model_list_path() -> Path:
    """Find ``assets/model_list.yaml`` by walking up from this module.

    Works both locally (``backend/assets/...``) and in the container (``/app/assets/...``).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "assets" / "model_list.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("assets/model_list.yaml not found")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_model_list(path: Path | None = None) -> list[dict]:
    path = path or _resolve_model_list_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = _expand(payload).get("models", [])
    if not models:
        raise ValueError(f"No models defined in {path}")
    return models

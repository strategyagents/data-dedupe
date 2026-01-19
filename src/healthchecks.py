from __future__ import annotations

from pathlib import Path

import httpx

from src.config import Config


def ensure_data_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {', '.join(missing)}")


def check_qdrant(url: str) -> None:
    endpoint = url.rstrip("/") + "/collections"
    try:
        response = httpx.get(endpoint, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("Qdrant is not reachable. Check QDRANT_URL.") from exc


def check_embedding_provider(config: Config) -> str:
    if config.openai_api_key:
        _check_openai(config.openai_api_key)
        return "openai"
    if config.ollama_endpoint:
        _check_ollama(config.ollama_endpoint)
        return "ollama"
    raise RuntimeError(
        "No embedding provider configured. Set OPENAI_API_KEY or OLLAMA_ENDPOINT."
    )


def _check_openai(api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = httpx.get(
            "https://api.openai.com/v1/models", headers=headers, timeout=10.0
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("OpenAI API is not reachable or API key is invalid.") from exc


def _check_ollama(endpoint: str) -> None:
    url = endpoint.rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("Ollama endpoint is not reachable.") from exc

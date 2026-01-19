from __future__ import annotations

import time
from typing import Callable

import httpx

from src.config import Config


def get_embedder() -> Callable[[list[str]], list[list[float]]]:
    config = Config.from_env()
    if config.openai_api_key:
        return _openai_embedder(config)
    if config.ollama_endpoint:
        return _ollama_embedder(config)
    raise RuntimeError(
        "No embedding provider configured. Set OPENAI_API_KEY or OLLAMA_ENDPOINT."
    )


def _openai_embedder(config: Config) -> Callable[[list[str]], list[list[float]]]:
    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": config.embed_model,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {"Authorization": f"Bearer {config.openai_api_key}"}
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        items = data.get("data", [])
        items.sort(key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in items]
        _validate_vectors(vectors)
        if len(vectors) != len(texts):
            raise ValueError("OpenAI embeddings response size mismatch.")
        return vectors

    return embed


def _ollama_embedder(config: Config) -> Callable[[list[str]], list[list[float]]]:
    endpoint = config.ollama_endpoint.rstrip("/")
    url = f"{endpoint}/api/embeddings"

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for text in texts:
                payload = {"model": config.embed_model, "prompt": text}
                vectors.append(_ollama_post_with_retry(client, url, payload))
        _validate_vectors(vectors)
        return vectors

    return embed


def _ollama_post_with_retry(
    client: httpx.Client, url: str, payload: dict[str, str]
) -> list[float]:
    attempts = 3
    for attempt in range(attempts):
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise ValueError("Ollama response missing 'embedding'.")
            return embedding
        except (httpx.HTTPError, ValueError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError("Failed to fetch Ollama embeddings.") from exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError("Failed to fetch Ollama embeddings.")


def _validate_vectors(vectors: list[list[float] | None]) -> None:
    if not vectors:
        raise ValueError("No embedding vectors returned.")
    first = vectors[0]
    if not isinstance(first, list) or not first:
        raise ValueError("Embedding vectors are empty or invalid.")
    dimension = len(first)
    for idx, vector in enumerate(vectors):
        if not isinstance(vector, list):
            raise ValueError(f"Embedding at index {idx} is not a list.")
        if len(vector) != dimension:
            raise ValueError("Embedding vectors have inconsistent dimensions.")

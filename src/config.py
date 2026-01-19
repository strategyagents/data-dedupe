import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    embed_model: str
    openai_api_key: str
    ollama_endpoint: str
    qdrant_url: str
    sim_threshold: float
    top_k: int
    collection_name: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            ollama_endpoint=os.getenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434"),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            sim_threshold=_get_env_float("SIM_THRESHOLD", 0.83),
            top_k=_get_env_int("TOP_K", 5),
            collection_name=os.getenv("COLLECTION_NAME", "companies"),
        )


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float") from exc


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc

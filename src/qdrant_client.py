from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Config


def client() -> QdrantClient:
    config = Config.from_env()
    return QdrantClient(url=config.qdrant_url)


def ensure_collection(name: str, dim: int) -> None:
    qdrant = client()
    if qdrant.collection_exists(name):
        info = qdrant.get_collection(name)
        existing_dim = _extract_vector_size(info.config.params.vectors)
        if existing_dim is not None and existing_dim != dim:
            qdrant.delete_collection(name)
        else:
            return
    qdrant.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )


def upsert_vectors(name: str, rows: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    if len(rows) != len(vectors):
        raise ValueError("Rows and vectors length mismatch.")
    points: list[models.PointStruct] = []
    for row, vector in zip(rows, vectors, strict=True):
        payload = {
            "id": row["id"],
            "company_name": row["company_name"],
        }
        points.append(models.PointStruct(id=row["id"], vector=vector, payload=payload))
    qdrant = client()
    qdrant.upsert(collection_name=name, points=points)


def nearest(name: str, top_k: int) -> list[dict[str, Any]]:
    qdrant = client()
    results: list[dict[str, Any]] = []
    scroll_result = qdrant.scroll(
        collection_name=name,
        with_payload=True,
        with_vectors=True,
        limit=10_000,
    )
    points = scroll_result[0]
    for point in points:
        if point.vector is None:
            continue
        response = qdrant.query_points(
            collection_name=name,
            query=point.vector,
            limit=top_k + 1,
            with_payload=True,
        )
        added = 0
        for neighbor in response.points:
            if neighbor.id == point.id:
                continue
            results.append(
                {
                    "id": point.id,
                    "company_name": point.payload.get("company_name") if point.payload else None,
                    "neighbor_id": neighbor.id,
                    "neighbor_name": neighbor.payload.get("company_name") if neighbor.payload else None,
                    "score": neighbor.score,
                }
            )
            added += 1
            if added >= top_k:
                break
    return results


def query_top_by_vector(
    name: str, vector: list[float], top_k: int = 1
) -> list[models.ScoredPoint]:
    qdrant = client()
    response = qdrant.query_points(
        collection_name=name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return list(response.points)


def _extract_vector_size(vectors: Any) -> int | None:
    if isinstance(vectors, models.VectorParams):
        return vectors.size
    if isinstance(vectors, dict):
        for value in vectors.values():
            size = _extract_vector_size(value)
            if size is not None:
                return size
    return None

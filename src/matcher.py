from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.normalize import normalize_name


def build_pairs(neighbor_results: list[dict[str, Any]]) -> list[tuple[Any, Any, float]]:
    pairs: dict[tuple[Any, Any], float] = {}
    for result in neighbor_results:
        id1 = result.get("id")
        id2 = result.get("neighbor_id")
        score = float(result.get("score", 0.0))
        if id1 is None or id2 is None or id1 == id2:
            continue
        key = _ordered_pair(id1, id2)
        current = pairs.get(key)
        if current is None or score > current:
            pairs[key] = score
    return [(id1, id2, score) for (id1, id2), score in pairs.items()]


def _ordered_pair(id1: Any, id2: Any) -> tuple[Any, Any]:
    if str(id1) <= str(id2):
        return (id1, id2)
    return (id2, id1)


def cluster_candidates(
    pairs: list[tuple[Any, Any, float]], threshold: float
) -> list[set[Any]]:
    uf = _UnionFind()
    for id1, id2, score in pairs:
        if score >= threshold:
            uf.union(id1, id2)
    return uf.clusters()


def choose_canonical(rows_in_cluster: list[dict[str, Any]]) -> str:
    cleaned = [normalize_name(row["company_name"]) for row in rows_in_cluster]
    cleaned.sort(key=lambda name: (len(name), name.lower()))
    return cleaned[0] if cleaned else ""


def dedupe_mapping(
    rows: list[dict[str, Any]], clusters: list[set[Any]]
) -> dict[Any, dict[str, Any]]:
    cluster_sets = [set(cluster) for cluster in clusters if cluster]
    seen_ids = {member for cluster in cluster_sets for member in cluster}
    missing_ids = [row["id"] for row in rows if row["id"] not in seen_ids]
    for missing_id in missing_ids:
        cluster_sets.append({missing_id})

    cluster_sets.sort(key=_cluster_sort_key)
    row_by_id = {row["id"]: row for row in rows}
    mapping: dict[Any, dict[str, Any]] = {}
    for index, cluster in enumerate(cluster_sets, start=1):
        cluster_rows = [row_by_id[member] for member in cluster if member in row_by_id]
        canonical = choose_canonical(cluster_rows)
        members = [
            {"id": row["id"], "company_name": row["company_name"]}
            for row in cluster_rows
        ]
        cluster_id = f"cluster_{index}"
        for member in cluster:
            mapping[member] = {
                "cluster_id": cluster_id,
                "canonical_name": canonical,
                "members": members,
            }
    return mapping


def _cluster_sort_key(cluster: set[Any]) -> tuple[str, int]:
    ordered = sorted((str(member) for member in cluster))
    return ("|".join(ordered), len(ordered))


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[Any, Any] = {}
        self._rank: dict[Any, int] = {}

    def find(self, item: Any) -> Any:
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0
        if self._parent[item] != item:
            self._parent[item] = self.find(self._parent[item])
        return self._parent[item]

    def union(self, left: Any, right: Any) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._rank[root_left] < self._rank[root_right]:
            self._parent[root_left] = root_right
        elif self._rank[root_left] > self._rank[root_right]:
            self._parent[root_right] = root_left
        else:
            self._parent[root_right] = root_left
            self._rank[root_left] += 1

    def clusters(self) -> list[set[Any]]:
        groups: dict[Any, set[Any]] = defaultdict(set)
        for item in self._parent:
            root = self.find(item)
            groups[root].add(item)
        return list(groups.values())

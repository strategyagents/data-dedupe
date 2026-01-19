from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any


def evaluate_if_available(
    gold_path: str | Path | None, mapping: dict[Any, dict[str, Any]]
) -> dict[str, float] | None:
    if gold_path is None:
        return None
    path = Path(gold_path)
    if not path.exists():
        print("no gold file found")
        return None

    gold_group = _load_gold_groups(path)

    predicted_pairs = _pairs_from_mapping(mapping)
    gold_pairs = _pairs_from_groups(gold_group)

    true_positive = len(predicted_pairs & gold_pairs)
    precision = _safe_divide(true_positive, len(predicted_pairs))
    recall = _safe_divide(true_positive, len(gold_pairs))
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_pairs": float(len(predicted_pairs)),
        "gold_pairs": float(len(gold_pairs)),
    }

    print(
        "Evaluation metrics: "
        f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
    )
    return metrics


def _load_gold_groups(path: Path) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "group_id" not in reader.fieldnames:
            raise ValueError("Gold file missing group_id column.")
        for row in reader:
            raw_id = (row.get("id") or "").strip()
            group_id = (row.get("group_id") or "").strip()
            if not raw_id or not group_id:
                continue
            try:
                parsed_id: Any = int(raw_id)
            except ValueError:
                parsed_id = raw_id
            groups.setdefault(group_id, []).append(parsed_id)
    return groups


def _pairs_from_mapping(mapping: dict[Any, dict[str, Any]]) -> set[tuple[Any, Any]]:
    grouped: dict[str, list[Any]] = {}
    for member_id, entry in mapping.items():
        cluster_id = entry.get("cluster_id")
        if cluster_id is None:
            continue
        grouped.setdefault(str(cluster_id), []).append(member_id)

    pairs: set[tuple[Any, Any]] = set()
    for members in grouped.values():
        for left, right in combinations(sorted(members, key=str), 2):
            pairs.add((left, right))
    return pairs


def _pairs_from_groups(groups: dict[str, list[Any]]) -> set[tuple[Any, Any]]:
    pairs: set[tuple[Any, Any]] = set()
    for members in groups.values():
        for left, right in combinations(sorted(members, key=str), 2):
            pairs.add((left, right))
    return pairs


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from src.config import Config
from src.evaluate import evaluate_if_available
from src.healthchecks import check_embedding_provider, check_qdrant, ensure_data_files
from src.loaders import load_companies
from src.matcher import build_pairs, cluster_candidates, dedupe_mapping
from src.normalize import normalize_name
from src.qdrant_client import (
    ensure_collection,
    nearest,
    query_top_by_vector,
    upsert_vectors,
)
from src.report import write_report
from src.embedder import get_embedder


def run_pipeline(
    *,
    config: Config,
    data_path: Path,
    report_path: Path,
    gold_path: Path | None = None,
    master_path: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    def log_step(message: str) -> None:
        elapsed = time.perf_counter() - start_time
        log(f"[{elapsed:6.2f}s] {message}")

    log_step("Running health checks")
    paths = [data_path]
    if master_path is not None:
        paths.append(master_path)
    ensure_data_files(paths)
    provider = check_embedding_provider(config)
    check_qdrant(config.qdrant_url)
    log_step(f"Health checks passed (provider: {provider})")

    log_step("Loading data")
    companies = load_companies(data_path)
    log(f"Loaded {len(companies)} companies from {data_path}.")
    for row in companies[:5]:
        normalized = normalize_name(row["company_name"])
        log(f"  id={row['id']} raw='{row['company_name']}' normalized='{normalized}'")

    log_step("Generating embeddings")
    embedder = get_embedder()
    names = [row["company_name"] for row in companies]
    vectors = embedder(names)

    pairs: list[tuple[int | str, int | str, float]] = []
    mapping: dict[int | str, dict[str, object]] = {}

    master_collection = None
    id_to_vector = {}
    if vectors:
        log(f"Generated {len(vectors)} embeddings with dimension {len(vectors[0])}.")
        id_to_vector = {row["id"]: vector for row, vector in zip(companies, vectors)}
        log_step("Upserting vectors into Qdrant")
        ensure_collection(config.collection_name, len(vectors[0]))
        upsert_vectors(config.collection_name, companies, vectors)
        log(f"Upserted {len(vectors)} vectors into '{config.collection_name}'.")

    if master_path is not None:
        log_step("Loading master list")
        master_rows = load_companies(master_path)
        log(f"Loaded {len(master_rows)} master rows from {master_path}.")
        master_names = [row["company_name"] for row in master_rows]
        log_step("Embedding master list")
        master_vectors = embedder(master_names)
        if master_vectors:
            master_collection = f"{config.collection_name}_master"
            ensure_collection(master_collection, len(master_vectors[0]))
            upsert_vectors(master_collection, master_rows, master_vectors)
            log(
                f"Upserted {len(master_vectors)} master vectors into '{master_collection}'."
            )

    if vectors:
        log_step("Searching nearest neighbors")
        neighbor_results = nearest(config.collection_name, config.top_k)
        log(f"Retrieved {len(neighbor_results)} neighbor records.")
        pairs = build_pairs(neighbor_results)
        log(f"Built {len(pairs)} candidate pairs.")

        top_pairs = sorted(pairs, key=lambda item: item[2], reverse=True)[:5]
        for id1, id2, score in top_pairs:
            log(f"  pair {id1} <-> {id2} score={score:.4f}")

        log_step("Clustering candidates")
        clusters = cluster_candidates(pairs, config.sim_threshold)
        mapping = dedupe_mapping(companies, clusters)
        if master_collection is not None and id_to_vector:
            _apply_master_canonicals(
                mapping, master_collection, id_to_vector, log=log
            )
        cluster_sizes = {}
        for entry in mapping.values():
            cluster_sizes[entry["cluster_id"]] = len(entry.get("members", []))
        non_trivial_count = sum(1 for size in cluster_sizes.values() if size > 1)
        log(f"Identified {len(cluster_sizes)} clusters; {non_trivial_count} non-trivial.")

    log_step("Evaluating against gold (if available)")
    metrics = evaluate_if_available(gold_path, mapping)

    log_step("Writing report")
    write_report(report_path, config, companies, pairs, mapping, metrics)
    log("report.html written.")

    return {
        "companies": companies,
        "pairs": pairs,
        "mapping": mapping,
        "metrics": metrics,
        "report_path": report_path,
    }


def _apply_master_canonicals(
    mapping: dict[int | str, dict[str, object]],
    master_collection: str,
    id_to_vector: dict[int | str, list[float]],
    *,
    log: Callable[[str], None],
) -> None:
    cluster_entries: dict[str, dict[str, object]] = {}
    for entry in mapping.values():
        cluster_id = entry.get("cluster_id")
        if cluster_id is None:
            continue
        cluster_entries.setdefault(str(cluster_id), entry)

    for cluster_id, entry in cluster_entries.items():
        members = entry.get("members", [])
        if not isinstance(members, list) or not members:
            continue
        representative = _choose_representative(members)
        vector = id_to_vector.get(representative["id"])
        if vector is None:
            continue
        matches = query_top_by_vector(master_collection, vector, top_k=1)
        if not matches:
            continue
        payload = matches[0].payload or {}
        master_name = payload.get("company_name")
        if not master_name:
            continue
        for member in members:
            member_id = member["id"]
            mapping[member_id]["canonical_name"] = master_name
        log(f"Cluster {cluster_id} canonical set to master: {master_name}")


def _choose_representative(members: list[dict[str, object]]) -> dict[str, object]:
    def sort_key(member: dict[str, object]) -> tuple[int, str, str]:
        name = normalize_name(str(member.get("company_name", "")))
        return (len(name), name.lower(), str(member.get("id")))

    return sorted(members, key=sort_key)[0]

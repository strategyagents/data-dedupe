from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from src.config import Config


def write_report(
    path: Path,
    config: Config,
    companies: list[dict[str, Any]],
    pairs: list[tuple[Any, Any, float]],
    mapping: dict[Any, dict[str, Any]],
    metrics: dict[str, float] | None = None,
) -> None:
    id_to_name = {row["id"]: row["company_name"] for row in companies}
    top_pairs = sorted(pairs, key=lambda item: item[2], reverse=True)[:25]

    cluster_map: dict[str, dict[str, Any]] = {}
    for entry in mapping.values():
        cluster_id = entry["cluster_id"]
        cluster = cluster_map.setdefault(
            cluster_id,
            {"canonical": entry["canonical_name"], "members": {}},
        )
        for member in entry.get("members", []):
            cluster["members"][member["id"]] = member["company_name"]

    cluster_rows = []
    for cluster_id in sorted(cluster_map.keys()):
        cluster = cluster_map[cluster_id]
        members = sorted(cluster["members"].items(), key=lambda item: str(item[0]))
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "canonical": cluster["canonical"],
                "members": members,
            }
        )

    deduped_count = len(cluster_rows) if cluster_rows else len(companies)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Embedding Data Cleansing Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 0.5rem; text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
    .muted {{ color: #627d98; }}
  </style>
</head>
<body>
  <h1>Embedding Data Cleansing Report</h1>

  <h2>Summary</h2>
  <ul>
    <li>Raw records: <strong>{len(companies)}</strong></li>
    <li>Deduped clusters: <strong>{deduped_count}</strong></li>
    <li>Threshold: <code>{config.sim_threshold}</code></li>
    <li>Top-K: <code>{config.top_k}</code></li>
    <li>Model: <code>{escape(config.embed_model)}</code></li>
  </ul>

  <h2>Top matched pairs</h2>
  <table>
    <thead>
      <tr>
        <th>ID 1</th>
        <th>Name 1</th>
        <th>ID 2</th>
        <th>Name 2</th>
        <th>Score</th>
      </tr>
    </thead>
    <tbody>
"""

    if top_pairs:
        for id1, id2, score in top_pairs:
            name1 = escape(str(id_to_name.get(id1, "")))
            name2 = escape(str(id_to_name.get(id2, "")))
            html += (
                "      <tr>"
                f"<td>{escape(str(id1))}</td>"
                f"<td>{name1}</td>"
                f"<td>{escape(str(id2))}</td>"
                f"<td>{name2}</td>"
                f"<td>{score:.4f}</td>"
                "</tr>\n"
            )
    else:
        html += (
            "      <tr><td colspan=\"5\" class=\"muted\">"
            "No pairs available.</td></tr>\n"
        )

    html += """    </tbody>
  </table>

  <h2>Clusters</h2>
  <table>
    <thead>
      <tr>
        <th>Cluster</th>
        <th>Canonical</th>
        <th>Members</th>
      </tr>
    </thead>
    <tbody>
"""

    if cluster_rows:
        for cluster in cluster_rows:
            members_html = ", ".join(
                f"{escape(str(member_id))}: {escape(member_name)}"
                for member_id, member_name in cluster["members"]
            )
            html += (
                "      <tr>"
                f"<td>{escape(cluster['cluster_id'])}</td>"
                f"<td>{escape(cluster['canonical'])}</td>"
                f"<td>{members_html}</td>"
                "</tr>\n"
            )
    else:
        html += (
            "      <tr><td colspan=\"3\" class=\"muted\">"
            "No clusters available.</td></tr>\n"
        )

    html += """    </tbody>
  </table>
"""

    if metrics:
        html += """\n  <h2>Evaluation (gold)</h2>\n  <ul>\n"""
        html += (
            f"    <li>Precision: <code>{metrics['precision']:.3f}</code></li>\n"
            f"    <li>Recall: <code>{metrics['recall']:.3f}</code></li>\n"
            f"    <li>F1: <code>{metrics['f1']:.3f}</code></li>\n"
            f"    <li>Predicted pairs: <code>{int(metrics['predicted_pairs'])}</code></li>\n"
            f"    <li>Gold pairs: <code>{int(metrics['gold_pairs'])}</code></li>\n"
        )
        html += "  </ul>\n"

    html += """</body>
</html>
"""

    path.write_text(html, encoding="utf-8")

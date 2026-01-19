import csv
from pathlib import Path
from typing import Any


def load_companies(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = (row.get("id") or "").strip()
            company_name = (row.get("company_name") or "").strip()
            if raw_id == "":
                continue
            parsed_id: int | str
            try:
                parsed_id = int(raw_id)
            except ValueError:
                parsed_id = raw_id
            rows.append({"id": parsed_id, "company_name": company_name})

    rows.sort(key=_sort_key)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
    value = row.get("id")
    if isinstance(value, int):
        return (0, f"{value:012d}")
    return (1, str(value))

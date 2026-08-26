"""Small parsing, calendar, and serialization helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def optional_float(value: str) -> float | None:
    stripped = value.strip()
    return None if not stripped else float(stripped)


def optional_int(value: str) -> int | None:
    stripped = value.strip()
    return None if not stripped else int(stripped)


def month_number(value: str) -> int:
    """Convert YYYYMM into an integer that increases by one each month."""
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f"Invalid monthly period: {value!r}")
    year = int(value[:4])
    month = int(value[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid monthly period: {value!r}")
    return year * 12 + month - 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


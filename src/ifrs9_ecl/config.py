"""Configuration loading and project path resolution."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project.yaml"


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project settings with paths resolved from the repository root."""

    raw: dict[str, Any]
    root: Path = PROJECT_ROOT

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        return (self.root / str(value)).resolve()

    @property
    def source_root(self) -> Path:
        data = self.raw["data"]
        configured = data.get("source_root")
        if configured:
            return Path(str(configured)).expanduser().resolve()
        environment_name = str(data.get("source_root_environment", "FREDDIE_DATA_ROOT"))
        environment_value = os.environ.get(environment_name)
        if environment_value:
            return Path(environment_value).expanduser().resolve()
        raise ValueError(
            f"Freddie Mac data root is not configured. Set {environment_name}, pass "
            "--data-root, or create config/local.yaml."
        )

    @property
    def phase0(self) -> dict[str, Any]:
        return deepcopy(self.raw["phase0"])

    def archive_path(self, vintage: str) -> Path:
        normalized = vintage.upper()
        if len(normalized) != 6 or normalized[4] != "Q":
            raise ValueError(f"Vintage must look like 2005Q1, received {vintage!r}")
        year = normalized[:4]
        quarter = normalized[5]
        if not year.isdigit() or quarter not in {"1", "2", "3", "4"}:
            raise ValueError(f"Vintage must look like 2005Q1, received {vintage!r}")
        return (
            self.source_root
            / f"historical_data_{year}"
            / f"historical_data_{normalized}.zip"
        )

    def ensure_output_directories(self) -> None:
        for key in ("processed_data", "artifacts", "figures"):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    data_root: str | Path | None = None,
) -> ProjectConfig:
    """Load the committed config, then optional local and command line overrides."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    local_path = config_path.with_name("local.yaml")
    if local_path.is_file():
        with local_path.open(encoding="utf-8") as handle:
            raw = _deep_merge(raw, yaml.safe_load(handle) or {})
    if data_root is not None:
        raw = _deep_merge(raw, {"data": {"source_root": str(data_root)}})
    required = {"project", "paths", "data", "phase0"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")
    return ProjectConfig(raw=raw)


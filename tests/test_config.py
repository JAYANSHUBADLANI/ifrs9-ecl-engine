"""Tests for deterministic project configuration."""

from pathlib import Path

import pytest

from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.utils import month_number


def test_archive_path_uses_year_folder(tmp_path: Path) -> None:
    config = ProjectConfig(
        raw={
            "paths": {},
            "data": {"source_root": str(tmp_path)},
            "phase0": {},
        }
    )
    assert config.archive_path("2005Q1") == (
        tmp_path / "historical_data_2005" / "historical_data_2005Q1.zip"
    )


@pytest.mark.parametrize("value", ["2005", "2005Q0", "2005Q5", "05Q1", "bad"])
def test_archive_path_rejects_bad_vintage(value: str, tmp_path: Path) -> None:
    config = ProjectConfig(
        raw={
            "paths": {},
            "data": {"source_root": str(tmp_path)},
            "phase0": {},
        }
    )
    with pytest.raises(ValueError):
        config.archive_path(value)


def test_month_number_is_consecutive_across_year_end() -> None:
    assert month_number("200601") - month_number("200512") == 1


@pytest.mark.parametrize("value", ["200500", "200513", "2005", "ABCDEF"])
def test_month_number_rejects_bad_period(value: str) -> None:
    with pytest.raises(ValueError):
        month_number(value)


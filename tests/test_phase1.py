"""Tests for Phase 1 segment and adjacency helpers."""

import pandas as pd

from ifrs9_ecl.phase1 import (
    _adjacent_pairs,
    _segmented_transition_rows,
    fico_band,
    ltv_band,
)


def test_credit_bands_preserve_unknown_values() -> None:
    fico = fico_band(pd.Series([650, 680, 720, 760, 9999, ""]))
    ltv = ltv_band(pd.Series([60, 75, 85, 95, 999, ""]))
    assert fico.tolist() == [
        "Below 660",
        "660 to 699",
        "700 to 739",
        "740 plus",
        "Unknown",
        "Unknown",
    ]
    assert ltv.tolist() == [
        "60 or below",
        "61 to 80",
        "81 to 90",
        "Above 90",
        "Unknown",
        "Unknown",
    ]


def test_segmented_pairs_exclude_calendar_gaps() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["a", "a", "a", "b", "b"],
            "reporting_month": [202001, 202002, 202004, 202001, 202002],
            "state": ["CURRENT", "30_DPD", "60_DPD", "CURRENT", "CURRENT"],
            "band": ["A", "A", "A", "B", "B"],
        }
    )
    pairs = _adjacent_pairs(panel)
    rows = _segmented_transition_rows(
        pairs,
        segment_column="band",
        segment_type="test",
        expansion_weight=5.0,
    )
    assert len(pairs) == 2
    assert rows["sample_count"].sum() == 2
    assert rows["weighted_count"].sum() == 10


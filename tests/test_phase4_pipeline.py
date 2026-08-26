"""Synthetic integration tests for Phase 4 staging and ECL."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.phase4 import (
    assemble_phase4_inputs,
    calculate_phase4,
    project_phase4_ead,
    run_phase4,
    scenarios_from_settings,
)


def _phase4_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["stage1", "relative", "absolute", "dpd30", "default"],
            "current_marginal_pd": [
                np.full(24, 0.0010),
                np.full(24, 0.0040),
                np.full(24, 0.0060),
                np.full(24, 0.0010),
                np.full(24, 0.0010),
            ],
            "origination_marginal_pd": [
                np.full(24, 0.0008),
                np.full(24, 0.0020),
                np.full(24, 0.0035),
                np.full(24, 0.0010),
                np.full(24, 0.0010),
            ],
            "lgd": [0.5, 0.5, 0.5, 0.5, 0.6],
            "ead": [100.0, 100.0, 100.0, 100.0, 80.0],
            "days_past_due": [0, 0, 0, 30, 0],
            "is_default": [False, False, False, False, True],
            "effective_annual_rate": [0.0] * 5,
            "vintage": ["A", "A", "B", "B", "B"],
        }
    )


def _config(tmp_path: Path, snapshot_path: Path) -> ProjectConfig:
    return ProjectConfig(
        root=tmp_path,
        raw={
            "project": {"name": "test"},
            "paths": {
                "processed_data": "data/processed",
                "artifacts": "artifacts",
                "figures": "reports/figures",
            },
            "data": {"source_root": str(tmp_path)},
            "phase0": {},
            "phase4": {
                "snapshot_path": str(snapshot_path),
                "reuse_completed": True,
            },
            "ifrs9": {
                "sicr_relative_lifetime_pd_ratio": 2.0,
                "sicr_absolute_lifetime_pd_increase": 0.05,
                "stage2_dpd_backstop": 30,
                "stage3_dpd_threshold": 90,
            },
            "scenarios": [
                {"name": "upside", "weight": 0.2, "hazard_multiplier": 0.75},
                {"name": "base", "weight": 0.6, "hazard_multiplier": 1.0},
                {"name": "downside", "weight": 0.2, "hazard_multiplier": 1.6},
            ],
        },
    )


def test_phase4_applies_both_sicr_tests_and_dpd_backstops() -> None:
    calculation = calculate_phase4(_phase4_inputs())
    loans = calculation.loan_results.set_index("loan_id")

    assert loans.loc["stage1", "stage"] == 1
    assert loans.loc["stage1", "ecl_horizon_months"] == 12
    assert loans.loc["stage1", "baseline_ecl"] == pytest.approx(0.6)
    assert loans.loc["relative", "stage"] == 2
    assert loans.loc["relative", "staging_reason"] == "relative_pd_sicr"
    assert loans.loc["relative", "ecl_horizon_months"] == 24
    assert loans.loc["absolute", "stage"] == 2
    assert loans.loc["absolute", "staging_reason"] == "absolute_pd_sicr"
    assert loans.loc["dpd30", "stage"] == 2
    assert loans.loc["dpd30", "staging_reason"] == "30_dpd_backstop"
    assert loans.loc["default", "stage"] == 3
    assert loans.loc["default", "baseline_ecl"] == pytest.approx(48.0)
    assert loans.loc["default", "probability_weighted_ecl"] == pytest.approx(48.0)
    assert "current_marginal_pd" not in loans.columns
    assert calculation.scenario_summary["weight"].sum() == pytest.approx(1.0)


def test_phase4_scenarios_shift_log_odds_in_expected_direction() -> None:
    calculation = calculate_phase4(_phase4_inputs().iloc[:1])
    loan = calculation.loan_results.iloc[0]

    assert loan["scenario_upside_ecl"] < loan["scenario_base_ecl"]
    assert loan["scenario_base_ecl"] < loan["scenario_downside_ecl"]
    scenarios = scenarios_from_settings(
        [
            {"name": "low", "weight": 1, "hazard_multiplier": 0.5},
            {"name": "high", "weight": 3, "hazard_multiplier": 2.0},
        ]
    )
    assert [scenario.weight for scenario in scenarios] == pytest.approx([0.25, 0.75])
    assert [scenario.hazard_shift for scenario in scenarios] == pytest.approx(
        [np.log(0.5), np.log(2.0)]
    )


def test_phase4_uses_audited_scheduled_ead_fallback_for_missing_scalar() -> None:
    inputs = _phase4_inputs().iloc[:1].copy()
    inputs["ead"] = np.nan
    inputs["current_actual_upb"] = 100.0
    inputs["remaining_term_months"] = 24

    result = calculate_phase4(inputs).loan_results.iloc[0]

    assert result["ead_method"] == "scheduled_amortization_with_snapshot_add_on"
    assert result["ead_at_reporting_date"] == pytest.approx(100.0)


def test_phase4_amortizes_principal_and_keeps_snapshot_nonprincipal_add_on() -> None:
    curve, add_on = project_phase4_ead(
        100.0,
        24,
        0.12,
        12,
        snapshot_ead=107.0,
    )

    assert add_on == pytest.approx(7.0)
    assert len(curve) == 12
    assert np.all(np.diff(curve) < 0.0)
    assert curve[0] < 107.0
    assert curve[-1] > 7.0


def test_phase4_adapter_accepts_long_pd_tables() -> None:
    snapshot = pd.DataFrame(
        {
            "loan_id": ["a", "b"],
            "days_past_due": [0, 0],
            "is_default": [False, False],
        }
    )
    current = pd.DataFrame(
        {
            "loan_id": ["a", "a", "b", "b"],
            "month": [1, 2, 1, 2],
            "marginal_pd": [0.01, 0.02, 0.03, 0.04],
        }
    )
    origination = current.assign(marginal_pd=current["marginal_pd"] / 2)
    lgd = pd.DataFrame({"loan_id": ["a", "b"], "predicted_lgd": [0.4, 0.5]})
    ead = pd.DataFrame({"loan_id": ["a", "b"], "ead_curve": [100.0, 80.0]})

    result = assemble_phase4_inputs(
        snapshot,
        current_pd=current,
        origination_pd=origination,
        lgd=lgd,
        ead=ead,
    )

    np.testing.assert_allclose(result.loc[0, "current_marginal_pd"], [0.01, 0.02])
    assert result["lgd"].tolist() == [0.4, 0.5]
    assert result["ead"].tolist() == [100.0, 80.0]


def test_run_phase4_writes_and_reuses_completed_checkpoint(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.parquet"
    _phase4_inputs().to_parquet(snapshot_path, index=False)
    config = _config(tmp_path, snapshot_path)

    first = run_phase4(config)
    second = run_phase4(config)

    assert first == second
    assert first["snapshot_loans"] == 5
    assert first["stage_counts"] == {"1": 1, "2": 3, "3": 1}
    for output in first["outputs"].values():
        assert Path(output).is_file()


def test_phase4_rejects_duplicate_snapshot_loans() -> None:
    duplicated = pd.concat([_phase4_inputs().iloc[:1]] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        calculate_phase4(duplicated)

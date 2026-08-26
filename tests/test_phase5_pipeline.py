"""Synthetic integration tests for the Phase 5 provision backtest."""

from pathlib import Path

import pandas as pd
import pytest

from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.phase5 import calculate_phase5, run_phase5


def _loans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["one", "two", "three"],
            "probability_weighted_ecl": [90.0, 20.0, 5.0],
            "effective_annual_rate": [0.21, 0.0, 0.0],
            "stage": [2, 1, 1],
            "ecl_horizon_months": [360, 12, 12],
            "vintage": ["A", "A", "B"],
        }
    )


def _losses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["one", "one", "two", "outside", "three", "one"],
            "reporting_month": [201212, 201312, 201301, 201301, 202604, 201401],
            "actual_loss": [999.0, 121.0, -10.0, 1000.0, 500.0, 0.0],
        }
    )


def _config(tmp_path: Path, phase4_path: Path, loss_path: Path) -> ProjectConfig:
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
            "modeling": {
                "backtest_snapshot_month": 201212,
                "performance_cutoff_month": 202603,
            },
            "phase5": {
                "phase4_loan_ecl_path": str(phase4_path),
                "realized_losses_path": str(loss_path),
                "segment_columns": ["stage"],
                "reuse_completed": True,
                "align_to_ecl_horizon": True,
            },
        },
    )


def test_phase5_filters_time_window_discounts_events_and_retains_signed_loss() -> None:
    calculation = calculate_phase5(
        _loans(),
        _losses(),
        snapshot_month=201212,
        cutoff_month=202603,
        segment_columns=["stage"],
    )
    detail = calculation.result.detail.set_index("loan_id")

    assert len(calculation.realized_loss_events) == 3
    assert detail.loc["one", "loss_event_count"] == 2
    assert detail.loc["one", "realized_loss"] == pytest.approx(121.0)
    assert detail.loc["one", "discounted_realized_loss"] == pytest.approx(100.0)
    assert detail.loc["two", "discounted_realized_loss"] == pytest.approx(-10.0)
    assert detail.loc["three", "discounted_realized_loss"] == 0.0
    assert calculation.result.overall.total_provision == pytest.approx(115.0)
    assert calculation.result.overall.total_discounted_realized_loss == pytest.approx(
        90.0
    )
    assert calculation.result.overall.provision_gap == pytest.approx(25.0)
    assert calculation.result.by_segment["observation_count"].sum() == 3


def test_phase5_aggregates_multiple_events_before_comparing_one_provision() -> None:
    losses = pd.DataFrame(
        {
            "loan_id": ["one", "one"],
            "reporting_month": [201301, 201302],
            "actual_loss": [10.0, 20.0],
        }
    )
    calculation = calculate_phase5(
        _loans().iloc[:1].assign(effective_annual_rate=0.0),
        losses,
        snapshot_month=201212,
        cutoff_month=201312,
    )

    assert calculation.result.overall.observation_count == 1
    assert calculation.result.overall.total_discounted_realized_loss == pytest.approx(
        30.0
    )


def test_phase5_aligns_stage1_losses_to_twelve_month_ecl_horizon() -> None:
    loans = pd.DataFrame(
        {
            "loan_id": ["stage1", "stage2"],
            "probability_weighted_ecl": [10.0, 10.0],
            "effective_annual_rate": [0.0, 0.0],
            "stage": [1, 2],
            "ecl_horizon_months": [12, 24],
        }
    )
    losses = pd.DataFrame(
        {
            "loan_id": ["stage1", "stage2"],
            "reporting_month": [201401, 201401],
            "actual_loss": [30.0, 40.0],
        }
    )

    aligned = calculate_phase5(
        loans,
        losses,
        snapshot_month=201212,
        cutoff_month=201412,
    )
    ultimate = calculate_phase5(
        loans,
        losses,
        snapshot_month=201212,
        cutoff_month=201412,
        align_to_ecl_horizon=False,
    )

    assert aligned.realized_loss_events["loan_id"].tolist() == ["stage2"]
    assert aligned.result.overall.total_discounted_realized_loss == pytest.approx(40.0)
    assert ultimate.result.overall.total_discounted_realized_loss == pytest.approx(70.0)


def test_phase5_stage3_uses_data_cutoff_when_modeled_horizon_is_zero() -> None:
    loan = _loans().iloc[[0]].assign(stage=3, ecl_horizon_months=0)
    losses = pd.DataFrame(
        {"loan_id": ["one"], "reporting_month": [201501], "actual_loss": [30.0]}
    )

    calculation = calculate_phase5(
        loan,
        losses,
        snapshot_month=201212,
        cutoff_month=201512,
    )

    assert len(calculation.realized_loss_events) == 1


def test_phase5_applies_sampling_weight_to_both_provision_and_loss() -> None:
    loans = (
        _loans()
        .iloc[:1]
        .assign(
            effective_annual_rate=0.0,
            exposure_weight=4.0,
        )
    )
    losses = pd.DataFrame(
        {"loan_id": ["one"], "reporting_month": [201301], "actual_loss": [30.0]}
    )

    calculation = calculate_phase5(
        loans,
        losses,
        snapshot_month=201212,
        cutoff_month=201312,
    )

    assert calculation.result.overall.total_provision == pytest.approx(360.0)
    assert calculation.result.overall.total_discounted_realized_loss == pytest.approx(
        120.0
    )
    detail = calculation.result.detail.iloc[0]
    assert detail["provision_unweighted"] == pytest.approx(90.0)
    assert detail["discounted_realized_loss_unweighted"] == pytest.approx(30.0)


def test_phase5_empty_loss_window_backtests_all_loans_against_zero() -> None:
    future = pd.DataFrame(
        {"loan_id": ["one"], "reporting_month": [202701], "actual_loss": [10.0]}
    )
    calculation = calculate_phase5(
        _loans(),
        future,
        snapshot_month=201212,
        cutoff_month=202603,
    )

    assert calculation.realized_loss_events.empty
    assert calculation.result.overall.total_discounted_realized_loss == 0.0
    assert calculation.result.overall.total_provision == pytest.approx(115.0)


def test_run_phase5_writes_and_reuses_completed_checkpoint(tmp_path: Path) -> None:
    phase4_path = tmp_path / "loan_ecl.parquet"
    loss_path = tmp_path / "losses.parquet"
    _loans().to_parquet(phase4_path, index=False)
    _losses().to_parquet(loss_path, index=False)
    config = _config(tmp_path, phase4_path, loss_path)

    first = run_phase5(config)
    second = run_phase5(config)

    assert first == second
    assert first["snapshot_loans"] == 3
    assert first["realized_loss_events"] == 3
    assert first["overall"]["provision_gap"] == pytest.approx(25.0)
    assert first["ultimate_stress_diagnostic"]["overall"][
        "total_discounted_realized_loss"
    ] == pytest.approx(90.0)
    for output in first["outputs"].values():
        assert Path(output).is_file()


def test_phase5_rejects_loss_at_or_before_cutoff_configuration() -> None:
    with pytest.raises(ValueError, match="after snapshot"):
        calculate_phase5(
            _loans(),
            _losses(),
            snapshot_month=201212,
            cutoff_month=201212,
        )

"""Tests for provision versus realized-loss backtesting."""

import math

import pandas as pd
import pytest

from ifrs9_ecl.backtest import (
    backtest_provisions,
    discount_realized_losses,
    summarize_backtest,
)


def test_realized_losses_are_discounted_using_effective_annual_rates() -> None:
    discounted = discount_realized_losses(
        [121.0, 100.0],
        [12.0, 0.0],
        [0.21, 0.50],
    )

    assert discounted.tolist() == pytest.approx([100.0, 100.0])


def test_overall_and_segment_summaries_reconcile() -> None:
    observations = pd.DataFrame(
        {
            "loan_id": [1, 2, 3],
            "segment": ["prime", "prime", "near_prime"],
            "provision": [60.0, 40.0, 20.0],
            "realized_loss": [50.0, 30.0, 30.0],
            "months_to_loss": [0.0, 0.0, 0.0],
        }
    )

    result = backtest_provisions(
        observations,
        segment_cols="segment",
        months_to_loss_col="months_to_loss",
    )
    segments = result.by_segment.set_index("segment")

    assert result.overall.total_provision == pytest.approx(120.0)
    assert result.overall.total_discounted_realized_loss == pytest.approx(110.0)
    assert result.overall.provision_gap == pytest.approx(10.0)
    assert result.overall.coverage_ratio == pytest.approx(120.0 / 110.0)
    assert result.overall.outcome == "over_provisioned"
    assert segments.loc["prime", "outcome"] == "over_provisioned"
    assert segments.loc["near_prime", "outcome"] == "under_provisioned"
    assert segments["total_provision"].sum() == pytest.approx(120.0)
    assert segments["total_discounted_realized_loss"].sum() == pytest.approx(110.0)


def test_backtest_calculates_error_metrics_at_observation_level() -> None:
    result = backtest_provisions(
        pd.DataFrame(
            {
                "provision": [12.0, 18.0],
                "realized_loss": [10.0, 20.0],
            }
        )
    )

    assert result.overall.provision_gap == pytest.approx(0.0)
    assert result.overall.outcome == "matched"
    assert result.overall.mean_gap == pytest.approx(0.0)
    assert result.overall.mean_absolute_error == pytest.approx(2.0)
    assert result.overall.root_mean_squared_error == pytest.approx(2.0)


def test_pre_discounted_loss_column_takes_precedence() -> None:
    observations = pd.DataFrame(
        {
            "provision": [100.0],
            "realized_loss": [999.0],
            "discounted_actual": [80.0],
        }
    )

    result = backtest_provisions(
        observations,
        discounted_loss_col="discounted_actual",
    )

    assert result.overall.total_discounted_realized_loss == pytest.approx(80.0)
    assert result.overall.provision_gap == pytest.approx(20.0)
    assert math.isnan(result.detail.loc[0, "discount_factor"])


def test_calendar_month_difference_can_supply_realization_timing() -> None:
    observations = pd.DataFrame(
        {
            "provision": [100.0],
            "realized_loss": [121.0],
            "reporting_date": ["2024-01-31"],
            "loss_date": ["2025-01-01"],
        }
    )

    result = backtest_provisions(
        observations,
        reporting_date_col="reporting_date",
        realized_loss_date_col="loss_date",
        effective_annual_rate=0.21,
    )

    assert result.overall.total_discounted_realized_loss == pytest.approx(100.0)
    assert result.overall.outcome == "matched"


def test_multiple_segment_columns_are_preserved_in_output() -> None:
    observations = pd.DataFrame(
        {
            "vintage": [2005, 2005, 2006],
            "stage": [1, 2, 1],
            "provision": [1.0, 2.0, 3.0],
            "realized_loss": [1.0, 1.0, 5.0],
        }
    )

    result = summarize_backtest(
        observations,
        segment_cols=["vintage", "stage"],
    )

    assert result.segment_columns == ("vintage", "stage")
    assert list(result.by_segment.columns[:2]) == ["vintage", "stage"]
    assert len(result.by_segment) == 3


def test_signed_realized_gain_is_retained_not_floored() -> None:
    result = backtest_provisions(pd.DataFrame({"provision": [0.0], "realized_loss": [-10.0]}))

    assert result.overall.total_discounted_realized_loss == pytest.approx(-10.0)
    assert result.overall.provision_gap == pytest.approx(10.0)


def test_zero_realized_loss_has_explicit_coverage_ratio_behavior() -> None:
    positive = backtest_provisions(pd.DataFrame({"provision": [1.0], "realized_loss": [0.0]}))
    zero = backtest_provisions(pd.DataFrame({"provision": [0.0], "realized_loss": [0.0]}))

    assert positive.overall.coverage_ratio == float("inf")
    assert math.isnan(zero.overall.coverage_ratio)


def test_input_frame_is_not_mutated() -> None:
    observations = pd.DataFrame({"provision": [10.0], "realized_loss": [8.0]})
    before = observations.copy(deep=True)

    backtest_provisions(observations)

    pd.testing.assert_frame_equal(observations, before)


@pytest.mark.parametrize(
    ("months", "rates"),
    [([-1.0], [0.0]), ([1.0], [-1.0]), ([float("nan")], [0.0])],
)
def test_invalid_discount_inputs_are_rejected(months: list[float], rates: list[float]) -> None:
    with pytest.raises(ValueError):
        discount_realized_losses([10.0], months, rates)


def test_missing_segment_or_timing_columns_fail_loudly() -> None:
    observations = pd.DataFrame({"provision": [10.0], "realized_loss": [8.0]})

    with pytest.raises(KeyError, match="segment"):
        backtest_provisions(observations, segment_cols="segment")
    with pytest.raises(ValueError, match="both reporting"):
        backtest_provisions(observations, reporting_date_col="reporting_date")


def test_empty_backtest_returns_zero_totals_and_empty_segment_table() -> None:
    observations = pd.DataFrame(columns=["provision", "realized_loss", "segment"])

    result = backtest_provisions(observations, segment_cols="segment")

    assert result.overall.observation_count == 0
    assert result.overall.total_provision == 0.0
    assert result.overall.total_discounted_realized_loss == 0.0
    assert result.by_segment.empty

"""Tests for monthly discounted ECL calculations."""

import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.ecl import (
    calculate_ecl,
    calculate_portfolio_ecl,
    discount_factors,
    validate_marginal_pd,
)
from ifrs9_ecl.staging import Stage


def test_stage_1_uses_only_first_12_marginal_default_months() -> None:
    marginal_pd = np.full(18, 0.01)
    lgd = np.linspace(0.40, 0.57, 18)
    ead = np.linspace(100.0, 83.0, 18)

    result = calculate_ecl(
        marginal_pd,
        lgd,
        ead,
        stage=1,
        effective_annual_rate=0.0,
    )

    expected = np.sum(marginal_pd[:12] * lgd[:12] * ead[:12])
    assert result.stage is Stage.STAGE_1
    assert result.method == "12_month_ecl"
    assert result.horizon_months == 12
    assert result.total_ecl == pytest.approx(expected)
    assert len(result.to_frame()) == 12


def test_stage_2_uses_lifetime_curve_and_effective_rate_discounting() -> None:
    marginal_pd = np.asarray([0.10, 0.20])
    lgd = np.asarray([0.50, 0.40])
    ead = np.asarray([100.0, 80.0])

    result = calculate_ecl(
        marginal_pd,
        lgd,
        ead,
        stage="Stage 2",
        effective_annual_rate=0.12,
    )

    expected_monthly = marginal_pd * lgd * ead
    expected_factors = np.power(1.12, -np.asarray([1.0, 2.0]) / 12.0)
    assert result.method == "lifetime_ecl"
    assert result.total_undiscounted_ecl == pytest.approx(expected_monthly.sum())
    assert result.total_ecl == pytest.approx(np.sum(expected_monthly * expected_factors))


def test_monthly_formula_uses_marginal_not_cumulative_pd() -> None:
    result = calculate_ecl([0.10, 0.20], 0.5, 100.0, stage=2)

    assert result.undiscounted_monthly_ecl.tolist() == pytest.approx([5.0, 10.0])
    assert result.total_ecl == pytest.approx(15.0)


def test_stage_3_immediate_default_uses_first_lgd_and_ead_without_discount() -> None:
    result = calculate_ecl(
        [0.01, 0.01],
        [0.6, 0.5],
        [100.0, 90.0],
        stage=3,
        effective_annual_rate=0.10,
    )

    assert result.method == "immediate_default"
    assert result.months.tolist() == [0]
    assert result.marginal_pd.tolist() == [1.0]
    assert result.total_ecl == pytest.approx(60.0)


def test_stage_3_cash_shortfall_proxy_can_be_discounted_from_recovery_month() -> None:
    result = calculate_ecl(
        None,
        None,
        None,
        stage=3,
        effective_annual_rate=0.21,
        stage3_cash_shortfall=50.0,
        stage3_cash_shortfall_month=12,
    )

    assert result.method == "cash_shortfall_proxy"
    assert result.horizon_months == 12
    assert result.total_ecl == pytest.approx(50.0 / 1.21)


def test_discount_factors_support_zero_and_negative_effective_rates() -> None:
    assert discount_factors([0, 12], 0.0).tolist() == [1.0, 1.0]
    assert discount_factors([12], -0.2)[0] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "curve",
    [[], [-0.01], [1.01], [0.6, 0.5], [float("nan")]],
)
def test_invalid_marginal_pd_curves_are_rejected(curve: list[float]) -> None:
    with pytest.raises((TypeError, ValueError), match="marginal_pd"):
        validate_marginal_pd(curve)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lgd": 1.1, "ead": 100.0},
        {"lgd": 0.5, "ead": -1.0},
        {"lgd": [0.5, 0.5, 0.5], "ead": 100.0},
    ],
)
def test_invalid_loss_terms_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        calculate_ecl([0.01, 0.01], stage=2, **kwargs)


def test_portfolio_ecl_supports_curve_cells_and_preserves_input() -> None:
    exposures = pd.DataFrame(
        {
            "marginal_pd": [[0.01] * 13, [0.20]],
            "lgd": [0.5, 0.75],
            "ead": [100.0, 80.0],
            "stage": [1, 3],
            "effective_annual_rate": [0.0, 0.0],
        }
    )
    before = exposures.copy(deep=True)

    result = calculate_portfolio_ecl(exposures)

    assert result["ecl"].tolist() == pytest.approx([6.0, 60.0])
    assert result["ecl_horizon_months"].tolist() == [12, 0]
    assert result["ecl_method"].tolist() == ["12_month_ecl", "immediate_default"]
    pd.testing.assert_frame_equal(exposures, before)

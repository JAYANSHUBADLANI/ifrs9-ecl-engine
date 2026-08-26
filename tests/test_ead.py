"""Tests for fixed-rate mortgage amortization and EAD components."""

import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.ead import (
    add_default_ead,
    calculate_default_ead,
    default_ead_components,
    fixed_rate_monthly_payment,
    project_amortizing_ead,
    project_scheduled_balances,
    scheduled_amortization_schedule,
    scheduled_balance_at_month,
)


def test_zero_rate_payment_and_balance_are_straight_line() -> None:
    assert fixed_rate_monthly_payment(120_000, 0.0, 120) == pytest.approx(1_000)
    assert scheduled_balance_at_month(120_000, 0.0, 120, 20) == pytest.approx(100_000)
    np.testing.assert_allclose(
        project_scheduled_balances(100, 0, 4), [100, 75, 50, 25, 0]
    )


def test_standard_fixed_rate_payment_and_maturity_balance() -> None:
    payment = fixed_rate_monthly_payment(100_000, 0.06, 360)

    assert payment == pytest.approx(599.550525, rel=1e-8)
    assert scheduled_balance_at_month(100_000, 0.06, 360, 1) == pytest.approx(
        99_900.449475, rel=1e-8
    )
    assert scheduled_balance_at_month(100_000, 0.06, 360, 360) == 0.0
    assert fixed_rate_monthly_payment(100_000, 6.0, 360, rate_in_percent=True) == pytest.approx(
        payment
    )
    assert fixed_rate_monthly_payment("100000", "6.0", "360", rate_in_percent=True) == (
        pytest.approx(payment)
    )


def test_amortization_schedule_reconciles_each_month_and_final_balance() -> None:
    schedule = scheduled_amortization_schedule(250_000, 0.045, 24)

    np.testing.assert_allclose(
        schedule["opening_balance"] - schedule["principal"],
        schedule["closing_balance"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        schedule["interest"] + schedule["principal"],
        schedule["scheduled_payment"],
        atol=1e-8,
    )
    assert schedule.iloc[-1]["closing_balance"] == 0.0
    assert schedule.iloc[:-1]["scheduled_payment"].nunique() == 1


def test_projection_beyond_maturity_stays_at_zero() -> None:
    balances = project_scheduled_balances(100, 0.0, 2, horizon_months=4)
    schedule = scheduled_amortization_schedule(100, 0.0, 2, horizon_months=4)

    np.testing.assert_allclose(balances, [100, 50, 0, 0, 0])
    assert schedule.loc[2:, "scheduled_payment"].eq(0).all()
    assert schedule.loc[2:, "closing_balance"].eq(0).all()


def test_default_ead_includes_non_interest_bearing_and_unpaid_amounts() -> None:
    components = default_ead_components(
        80_000,
        non_interest_bearing_upb=5_000,
        delinquent_accrued_interest=1_200,
        other_unpaid_amounts=300,
        undrawn_commitment=2_000,
        credit_conversion_factor=0.5,
    )

    assert components.undrawn_ead == 1_000
    assert components.total_ead == 87_500
    assert calculate_default_ead(
        80_000,
        non_interest_bearing_upb=5_000,
        delinquent_accrued_interest=1_200,
        other_unpaid_amounts=300,
        undrawn_commitment=2_000,
        credit_conversion_factor=0.5,
    ) == pytest.approx(87_500)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"non_interest_bearing_upb": -1}, "non_interest_bearing_upb"),
        ({"delinquent_accrued_interest": float("nan")}, "finite"),
        ({"credit_conversion_factor": 1.1}, "between zero and one"),
    ],
)
def test_default_ead_rejects_invalid_components(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        calculate_default_ead(100, **kwargs)


def test_projected_ead_adds_scalar_and_month_specific_components() -> None:
    projection = project_amortizing_ead(
        100,
        0.0,
        2,
        horizon_months=2,
        non_interest_bearing_upb=10,
        delinquent_accrued_interest=[0, 2, 4],
        other_unpaid_amounts=1,
    )

    np.testing.assert_allclose(projection["interest_bearing_upb"], [100, 50, 0])
    np.testing.assert_allclose(projection["default_ead"], [111, 63, 15])
    assert projection["month"].tolist() == [0, 1, 2]


def test_projected_ead_requires_one_component_value_per_projection_month() -> None:
    with pytest.raises(ValueError, match="must contain 4 values"):
        project_amortizing_ead(
            100,
            0.0,
            3,
            horizon_months=3,
            non_interest_bearing_upb=[1, 2],
        )


def test_add_default_ead_uses_explicit_components_without_mutating_input() -> None:
    data = pd.DataFrame(
        {
            "current_interest_bearing_upb": [100.0, 80.0],
            "current_non_interest_bearing_upb": [10.0, 5.0],
            "delinquent_accrued_interest": [2.0, 0.0],
        }
    )
    before = data.copy(deep=True)

    result = add_default_ead(data)

    assert result["default_ead"].tolist() == [112.0, 85.0]
    pd.testing.assert_frame_equal(data, before)


def test_invalid_term_rate_and_horizon_fail_loudly() -> None:
    with pytest.raises(ValueError, match="positive"):
        fixed_rate_monthly_payment(100, 0.05, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        fixed_rate_monthly_payment(100, -0.01, 12)
    with pytest.raises(TypeError, match="integer"):
        project_scheduled_balances(100, 0.05, 12.5)

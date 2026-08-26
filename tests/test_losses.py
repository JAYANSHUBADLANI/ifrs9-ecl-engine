"""Tests for Release 47 actual loss reconciliation."""

from decimal import Decimal

import pytest

from ifrs9_ecl.losses import reconcile_actual_loss


def test_reconciles_positive_loss_without_double_counting_expense_details() -> None:
    result = reconcile_actual_loss(
        {
            "actual_loss": "3714.66",
            "zero_balance_removal_upb": "77813.46",
            "net_sales_proceeds": "-89084.95",
            "delinquent_accrued_interest": "10991.15",
            "total_expenses": "3995.00",
            "mi_recoveries": "",
            "non_mi_recoveries": None,
            "legal_costs": "30.00",
            "maintenance_and_preservation_costs": "0.00",
            "taxes_and_insurance": "0.00",
            "miscellaneous_expenses": "3965.00",
        }
    )

    assert result.recomputed_loss == Decimal("3714.66")
    assert result.difference == Decimal("0.00")
    assert result.is_within_tolerance is True
    assert result.status == "reconciled"
    assert set(result.components) == {
        "zero_balance_removal_upb",
        "net_sales_proceeds",
        "delinquent_accrued_interest",
        "total_expenses",
        "mi_recoveries",
        "non_mi_recoveries",
    }


def test_reconciles_negative_gain() -> None:
    result = reconcile_actual_loss(
        {
            "actual_loss": "-48.00",
            "zero_balance_removal_upb": "100.00",
            "net_sales_proceeds": "-150.00",
            "delinquent_accrued_interest": " ",
            "total_expenses": "10.00",
            "mi_recoveries": "-5.00",
            "non_mi_recoveries": "-3.00",
        }
    )

    assert result.actual_loss == Decimal("-48.00")
    assert result.recomputed_loss == Decimal("-48.00")
    assert result.is_within_tolerance is True


@pytest.mark.parametrize("actual_loss", [None, "", " "])
def test_null_actual_loss_skips_component_recomputation(actual_loss: object) -> None:
    result = reconcile_actual_loss(
        {
            "actual_loss": actual_loss,
            "net_sales_proceeds": "not parsed when actual loss is absent",
        }
    )

    assert result.actual_loss is None
    assert result.recomputed_loss is None
    assert result.difference is None
    assert result.absolute_difference is None
    assert result.is_within_tolerance is None
    assert result.status == "actual_loss_missing"
    assert result.components == {}


def test_reports_outside_tolerance() -> None:
    result = reconcile_actual_loss(
        {
            "actual_loss": "10.02",
            "zero_balance_removal_upb": "10.00",
        },
        tolerance="0.01",
    )

    assert result.difference == Decimal("-0.02")
    assert result.absolute_difference == Decimal("0.02")
    assert result.is_within_tolerance is False
    assert result.status == "outside_tolerance"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actual_loss", "not-a-number"),
        ("net_sales_proceeds", "U"),
        ("total_expenses", float("inf")),
    ],
)
def test_rejects_invalid_nonnumeric_values(field: str, value: object) -> None:
    record: dict[str, object] = {"actual_loss": "0.00", field: value}

    with pytest.raises(ValueError, match=field):
        reconcile_actual_loss(record)


@pytest.mark.parametrize("tolerance", ["bad", "-0.01"])
def test_rejects_invalid_tolerance(tolerance: str) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        reconcile_actual_loss({"actual_loss": "0.00"}, tolerance=tolerance)

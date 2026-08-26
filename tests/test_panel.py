"""Tests for the streamed monthly panel writer."""

from pathlib import Path

import pyarrow.parquet as pq

from ifrs9_ecl.panel import build_panel_record, write_monthly_panel
from ifrs9_ecl.states import CURRENT, DEFAULTED


def origination(loan_id: str) -> dict[str, str]:
    return {
        "loan_id": loan_id,
        "first_payment_date": "200502",
        "classic_fico": "700",
    }


def performance(
    loan_id: str,
    month: str,
    delinquency: str = "00",
    zero_balance: str = "",
) -> dict[str, str]:
    return {
        "loan_id": loan_id,
        "reporting_month": month,
        "delinquency_status": delinquency,
        "zero_balance_code": zero_balance,
        "actual_loss": "",
    }


def test_build_panel_record_marks_cycle_and_terminal_state() -> None:
    row = build_panel_record(
        performance("a", "200501", "00", "02"), origination("a"), "2005Q1"
    )
    assert row["state"] == DEFAULTED
    assert row["is_pre_may_2019_cycle"] is True
    assert row["is_before_first_payment_cycle"] is True


def test_writer_streams_joined_rows_and_diagnostics(tmp_path: Path) -> None:
    rows = [
        performance("a", "200501"),
        performance("a", "200502", "01"),
        performance("b", "201905"),
    ]
    output = tmp_path / "panel.parquet"
    diagnostics = write_monthly_panel(
        iter(rows),
        {"a": origination("a"), "b": origination("b")},
        output,
        vintage="2005Q1",
        batch_rows=2,
    )
    stored = pq.read_table(output).to_pandas()
    assert len(stored) == 3
    assert stored.loc[0, "state"] == CURRENT
    assert stored.loc[0, "orig_classic_fico"] == "700"
    assert diagnostics.rows_written == 3
    assert len(diagnostics.loan_ids) == 2
    assert diagnostics.pre_may_2019_rows == 2
    assert diagnostics.may_2019_or_later_rows == 1
    assert diagnostics.before_first_payment_rows == 1


def test_actual_loss_reconciliation_is_counted(tmp_path: Path) -> None:
    row = performance("a", "201001", "00", "02")
    row.update(
        {
            "actual_loss": "3714.66",
            "zero_balance_removal_upb": "77813.46",
            "net_sales_proceeds": "-89084.95",
            "delinquent_accrued_interest": "10991.15",
            "total_expenses": "3995.00",
            "mi_recoveries": "0.00",
            "non_mi_recoveries": "0.00",
        }
    )
    diagnostics = write_monthly_panel(
        [row],
        {"a": origination("a")},
        tmp_path / "panel.parquet",
        vintage="2005Q1",
    )
    assert diagnostics.actual_loss_rows == 1
    assert diagnostics.actual_loss_reconciled_rows == 1
    assert diagnostics.actual_loss_outside_tolerance_rows == 0
    assert diagnostics.maximum_actual_loss_difference == 0


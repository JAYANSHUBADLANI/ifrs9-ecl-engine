from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ifrs9_ecl.archive import (
    ArchiveLayoutError,
    LoanOrderingError,
    RowWidthError,
    iter_member_rows,
    iter_performance_rows,
    locate_vintage_archive,
    read_origination_sample,
    resolve_archive_members,
)
from ifrs9_ecl.schemas import (
    ORIGINATION_COLUMNS,
    ORIGINATION_SCHEMA,
    PERFORMANCE_COLUMNS,
    PERFORMANCE_SCHEMA,
)


def _raw_values(schema, **overrides: str) -> list[str]:
    values = [""] * schema.width
    for column, value in overrides.items():
        values[schema.index(column)] = value
    return values


def _line(values: list[str]) -> str:
    return "|".join(values) + "\n"


def _write_archive(
    path: Path,
    *,
    origination_rows: list[list[str]] | None = None,
    performance_rows: list[list[str]] | None = None,
    origination_name: str = "orig_2005Q1.txt",
    performance_name: str = "perf_2005Q1.txt",
    extra_members: dict[str, str] | None = None,
) -> Path:
    origination_rows = origination_rows or []
    performance_rows = performance_rows or []
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            origination_name,
            "".join(_line(values) for values in origination_rows),
        )
        archive.writestr(
            performance_name,
            "".join(_line(values) for values in performance_rows),
        )
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
    return path


def _origination_row(loan_id: str, score: str = "700") -> list[str]:
    return _raw_values(
        ORIGINATION_SCHEMA,
        classic_fico=score,
        first_payment_date="200505",
        loan_id=loan_id,
        vantage_score_4_0="9999",
    )


def _performance_row(
    loan_id: str,
    reporting_month: str,
    *,
    delinquency_status: str = "00",
    bankruptcy_cramdown_costs: str = "",
) -> list[str]:
    return _raw_values(
        PERFORMANCE_SCHEMA,
        loan_id=loan_id,
        reporting_month=reporting_month,
        current_actual_upb="100000.00",
        delinquency_status=delinquency_status,
        zero_balance_code="",
        bankruptcy_cramdown_costs=bankruptcy_cramdown_costs,
    )


def test_release_47_schema_positions_are_exact() -> None:
    assert ORIGINATION_COLUMNS == (
        "classic_fico",
        "first_payment_date",
        "first_time_homebuyer_indicator",
        "maturity_date",
        "msa_or_metropolitan_division",
        "mi_percentage",
        "number_of_units",
        "occupancy_status",
        "original_cltv",
        "original_dti",
        "original_upb",
        "original_ltv",
        "original_interest_rate",
        "channel",
        "prepayment_penalty_indicator",
        "amortization_type",
        "property_state",
        "property_type",
        "postal_code",
        "loan_id",
        "loan_purpose",
        "original_loan_term",
        "number_of_borrowers",
        "seller_name",
        "super_conforming_flag",
        "pre_harp_loan_id",
        "special_eligibility_program",
        "harp_indicator",
        "property_valuation_method",
        "interest_only_indicator",
        "vantage_score_4_0",
    )
    assert PERFORMANCE_COLUMNS == (
        "loan_id",
        "reporting_month",
        "current_actual_upb",
        "delinquency_status",
        "loan_age",
        "remaining_months_to_legal_maturity",
        "defect_settlement_date",
        "modification_flag",
        "zero_balance_code",
        "zero_balance_effective_date",
        "current_interest_rate",
        "current_non_interest_bearing_upb",
        "ddlpi",
        "mi_recoveries",
        "net_sales_proceeds",
        "non_mi_recoveries",
        "total_expenses",
        "legal_costs",
        "maintenance_and_preservation_costs",
        "taxes_and_insurance",
        "miscellaneous_expenses",
        "actual_loss",
        "cumulative_modification_costs",
        "interest_rate_step_indicator",
        "payment_deferral_flag",
        "estimated_ltv",
        "zero_balance_removal_upb",
        "delinquent_accrued_interest",
        "delinquency_due_to_disaster",
        "borrower_assistance_plan",
        "current_period_modification_costs",
        "current_interest_bearing_upb",
        "mi_cancellation_indicator",
        "servicer_name",
        "bankruptcy_cramdown_costs",
    )
    assert ORIGINATION_SCHEMA.width == 31
    assert ORIGINATION_SCHEMA.index("loan_id") == 19
    assert PERFORMANCE_SCHEMA.width == 35
    assert PERFORMANCE_SCHEMA.index("actual_loss") == 21
    assert PERFORMANCE_SCHEMA.index("bankruptcy_cramdown_costs") == 34


def test_resolve_archive_members_validates_release_47_layout(tmp_path: Path) -> None:
    archive_path = _write_archive(tmp_path / "quarter.zip")

    members = resolve_archive_members(archive_path)

    assert members.vintage == "2005Q1"
    assert members.origination == "orig_2005Q1.txt"
    assert members.performance == "perf_2005Q1.txt"


def test_resolve_archive_members_rejects_extra_files(tmp_path: Path) -> None:
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        extra_members={"notes.txt": "not source data"},
    )

    with pytest.raises(ArchiveLayoutError, match="expected exactly 2"):
        resolve_archive_members(archive_path)


def test_resolve_archive_members_rejects_mixed_vintages(tmp_path: Path) -> None:
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        performance_name="perf_2005Q2.txt",
    )

    with pytest.raises(ArchiveLayoutError, match="member vintages differ"):
        resolve_archive_members(archive_path)


def test_stream_preserves_blank_trailing_performance_field(tmp_path: Path) -> None:
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        origination_rows=[_origination_row("F05Q10000001")],
        performance_rows=[_performance_row("F05Q10000001", "200504")],
    )

    stream = iter_performance_rows(archive_path)
    rows = list(stream)

    assert len(rows) == 1
    assert len(rows[0].values) == 35
    assert rows[0]["loan_id"] == "F05Q10000001"
    assert rows[0]["delinquency_status"] == "00"
    assert rows[0]["bankruptcy_cramdown_costs"] == ""
    assert rows[0].values[-1] == ""
    assert stream.diagnostics.physical_rows_scanned == 1
    assert stream.diagnostics.selected_rows_yielded == 1
    assert stream.diagnostics.completed_archive_scan is True


def test_member_stream_rejects_a_missing_trailing_field(tmp_path: Path) -> None:
    valid = _performance_row("F05Q10000001", "200504")
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        performance_rows=[valid[:-1]],
    )

    with pytest.raises(RowWidthError) as error:
        list(iter_member_rows(archive_path, "perf_2005Q1.txt", PERFORMANCE_SCHEMA))

    assert error.value.row_number == 1
    assert error.value.expected_width == 35
    assert error.value.actual_width == 34


def test_origination_sample_is_a_deterministic_prefix(tmp_path: Path) -> None:
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        origination_rows=[
            _origination_row("F05Q10000001", "701"),
            _origination_row("F05Q10000002", "702"),
            _origination_row("F05Q10000003", "703"),
        ],
    )

    first = read_origination_sample(archive_path, limit=2)
    second = read_origination_sample(archive_path, limit=2)

    assert first.loan_ids == ("F05Q10000001", "F05Q10000002")
    assert first.loan_ids == second.loan_ids
    assert [row["classic_fico"] for row in first.rows] == ["701", "702"]
    assert first.loan_id_set == frozenset(first.loan_ids)


def test_performance_prefix_scan_can_stop_after_all_selected_loans(tmp_path: Path) -> None:
    loan_1 = "F05Q10000001"
    loan_2 = "F05Q10000002"
    loan_3 = "F05Q10000003"
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        origination_rows=[
            _origination_row(loan_1),
            _origination_row(loan_2),
            _origination_row(loan_3),
        ],
        performance_rows=[
            _performance_row(loan_1, "200504"),
            _performance_row(loan_1, "200505"),
            _performance_row(loan_2, "200504"),
            _performance_row(loan_3, "200504"),
        ],
    )
    sample = read_origination_sample(archive_path, limit=2)

    stream = iter_performance_rows(
        archive_path,
        sample.loan_ids,
        stop_after_selected=True,
    )
    rows = list(stream)

    assert [row["loan_id"] for row in rows] == [loan_1, loan_1, loan_2]
    assert stream.diagnostics.physical_rows_scanned == 4
    assert stream.diagnostics.selected_rows_yielded == 3
    assert stream.diagnostics.selected_loans_found == 2
    assert stream.diagnostics.ordering_checks == 3
    assert stream.diagnostics.loan_order_valid is True
    assert stream.diagnostics.early_stop_occurred is True
    assert stream.diagnostics.completed_archive_scan is False
    assert stream.diagnostics.selection_complete is True


def test_disabling_order_validation_also_disables_early_stop(tmp_path: Path) -> None:
    loan_1 = "F05Q10000001"
    loan_2 = "F05Q10000002"
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        performance_rows=[
            _performance_row(loan_1, "200504"),
            _performance_row(loan_2, "200504"),
        ],
    )

    stream = iter_performance_rows(
        archive_path,
        [loan_1],
        stop_after_selected=True,
        validate_loan_order=False,
    )
    rows = list(stream)

    assert [row["loan_id"] for row in rows] == [loan_1]
    assert stream.diagnostics.physical_rows_scanned == 2
    assert stream.diagnostics.ordering_checks == 0
    assert stream.diagnostics.loan_order_valid is None
    assert stream.diagnostics.early_stop_occurred is False
    assert stream.diagnostics.completed_archive_scan is True


def test_performance_stream_detects_nonmonotonic_loan_ids(tmp_path: Path) -> None:
    archive_path = _write_archive(
        tmp_path / "quarter.zip",
        performance_rows=[
            _performance_row("F05Q10000002", "200504"),
            _performance_row("F05Q10000001", "200504"),
        ],
    )

    stream = iter_performance_rows(archive_path)
    with pytest.raises(LoanOrderingError, match="not nondecreasing"):
        list(stream)

    assert stream.diagnostics.physical_rows_scanned == 2
    assert stream.diagnostics.selected_rows_yielded == 1
    assert stream.diagnostics.ordering_checks == 1
    assert stream.diagnostics.loan_order_valid is False
    assert stream.diagnostics.completed_archive_scan is False


def test_locate_vintage_archive_supports_dataset_root_and_year_root(tmp_path: Path) -> None:
    year_root = tmp_path / "historical_data_2005"
    year_root.mkdir()
    archive_path = _write_archive(year_root / "historical_data_2005Q1.zip")

    assert locate_vintage_archive(tmp_path, "2005Q1") == archive_path.resolve()
    assert locate_vintage_archive(year_root, "2005Q1") == archive_path.resolve()


@pytest.mark.parametrize("bad_vintage", ["2005", "2005Q0", "2005Q5", "05Q1"])
def test_locate_vintage_archive_rejects_bad_vintage(
    tmp_path: Path,
    bad_vintage: str,
) -> None:
    with pytest.raises(ValueError, match="Invalid vintage"):
        locate_vintage_archive(tmp_path, bad_vintage)


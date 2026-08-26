"""Small ZIP integration test for the full Phase 0 entry point."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.phase0 import run_phase0
from ifrs9_ecl.schemas import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS


def raw_row(columns: tuple[str, ...], **values: str) -> str:
    return "|".join(values.get(column, "") for column in columns)


def write_fixture_archive(root: Path) -> None:
    directory = root / "source" / "historical_data_2005"
    directory.mkdir(parents=True)
    archive_path = directory / "historical_data_2005Q1.zip"
    origination = [
        raw_row(
            ORIGINATION_COLUMNS,
            loan_id="F05Q10000001",
            first_payment_date="200501",
            classic_fico="700",
        ),
        raw_row(
            ORIGINATION_COLUMNS,
            loan_id="F05Q10000002",
            first_payment_date="200501",
            classic_fico="680",
        ),
    ]
    performance = [
        raw_row(
            PERFORMANCE_COLUMNS,
            loan_id="F05Q10000001",
            reporting_month="200501",
            delinquency_status="00",
        ),
        raw_row(
            PERFORMANCE_COLUMNS,
            loan_id="F05Q10000001",
            reporting_month="200502",
            delinquency_status="01",
        ),
        raw_row(
            PERFORMANCE_COLUMNS,
            loan_id="F05Q10000001",
            reporting_month="200503",
            delinquency_status="00",
            zero_balance_code="02",
            actual_loss="10.00",
            zero_balance_removal_upb="100.00",
            net_sales_proceeds="-90.00",
        ),
        raw_row(
            PERFORMANCE_COLUMNS,
            loan_id="F05Q10000002",
            reporting_month="200501",
            delinquency_status="00",
        ),
        raw_row(
            PERFORMANCE_COLUMNS,
            loan_id="F05Q10000002",
            reporting_month="200502",
            delinquency_status="00",
            zero_balance_code="01",
        ),
    ]
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("orig_2005Q1.txt", "\n".join(origination) + "\n")
        archive.writestr("perf_2005Q1.txt", "\n".join(performance) + "\n")


def test_phase0_runs_end_to_end_on_headerless_release47_zip(tmp_path: Path) -> None:
    write_fixture_archive(tmp_path)
    config = ProjectConfig(
        root=tmp_path,
        raw={
            "paths": {
                "processed_data": "data/processed",
                "artifacts": "artifacts",
                "figures": "reports/figures",
            },
            "data": {"source_root": str(tmp_path / "source")},
            "phase0": {
                "vintage": "2005Q1",
                "maximum_loans": 2,
                "parquet_batch_rows": 2,
            },
        },
    )
    summary = run_phase0(config)
    assert summary["panel"]["rows_written"] == 5
    assert summary["panel"]["loan_count"] == 2
    assert summary["performance_scan"]["selection_complete"] is True
    assert summary["actual_loss_validation_passed"] is True
    assert Path(summary["outputs"]["monthly_panel"]).is_file()
    assert Path(summary["outputs"]["probabilities"]).is_file()
    assert Path(summary["outputs"]["figure"]).is_file()


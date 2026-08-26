"""Real-data smoke test for schema, panel, loss, and transition mechanics."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ifrs9_ecl.archive import (
    iter_performance_rows,
    read_origination_sample,
    resolve_archive_members,
)
from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.panel import write_monthly_panel
from ifrs9_ecl.reporting import save_transition_outputs
from ifrs9_ecl.schemas import (
    ORIGINATION_COLUMNS,
    ORIGINATION_SCHEMA,
    PERFORMANCE_SCHEMA,
    RELEASE_NUMBER,
)
from ifrs9_ecl.transitions import estimate_transitions
from ifrs9_ecl.utils import utc_timestamp, write_json


def _archive_audit(archive_path: Path) -> dict[str, Any]:
    members = resolve_archive_members(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        member_rows = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_rows.append(
                {
                    "name": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "encrypted": bool(info.flag_bits & 0x1),
                }
            )
    return {
        "release": RELEASE_NUMBER,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "vintage": members.vintage,
        "origination_member": members.origination,
        "performance_member": members.performance,
        "origination_expected_fields": ORIGINATION_SCHEMA.width,
        "performance_expected_fields": PERFORMANCE_SCHEMA.width,
        "members": member_rows,
    }


def _write_origination_sample(sample: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(column, pa.string()) for column in ORIGINATION_COLUMNS])
    table = pa.Table.from_pylist([row.as_dict() for row in sample.rows], schema=schema)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    pq.write_table(
        table,
        temporary_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    temporary_path.replace(path)


def _state_distribution(panel_path: Path) -> pd.DataFrame:
    state = pq.read_table(panel_path, columns=["state"]).column("state").to_pylist()
    counts = Counter(state)
    total = sum(counts.values())
    return pd.DataFrame(
        [
            {"state": state_name, "rows": count, "share": count / total}
            for state_name, count in sorted(counts.items())
        ]
    )


def run_phase0(
    config: ProjectConfig,
    *,
    vintage: str | None = None,
    maximum_loans: int | None = None,
) -> dict[str, Any]:
    """Run the bounded real-data smoke test and persist its evidence."""
    settings = config.phase0
    selected_vintage = str(vintage or settings["vintage"]).upper()
    loan_limit = int(maximum_loans or settings["maximum_loans"])
    if loan_limit < 1:
        raise ValueError("Phase 0 maximum_loans must be at least one")
    batch_rows = int(settings.get("parquet_batch_rows", 50_000))
    config.ensure_output_directories()
    archive_path = config.archive_path(selected_vintage)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Quarterly archive not found: {archive_path}")

    artifact_directory = config.path("artifacts")
    processed_directory = config.path("processed_data")
    figure_directory = config.path("figures")
    audit_directory = artifact_directory / "data_audit"
    table_directory = artifact_directory / "tables"
    prefix = f"phase0_{selected_vintage.lower()}"
    audit_path = audit_directory / f"{prefix}_archive.json"
    origination_path = processed_directory / f"{prefix}_origination.parquet"
    panel_path = processed_directory / f"{prefix}_panel.parquet"
    summary_path = artifact_directory / f"{prefix}_summary.json"
    state_path = table_directory / f"{prefix}_state_distribution.csv"

    archive_audit = _archive_audit(archive_path)
    write_json(archive_audit, audit_path)
    sample = read_origination_sample(archive_path, loan_limit)
    if len(sample.rows) != loan_limit:
        raise RuntimeError(
            f"Requested {loan_limit} origination rows but archive supplied {len(sample.rows)}"
        )
    _write_origination_sample(sample, origination_path)
    origination_lookup = {row["loan_id"]: row for row in sample.rows}

    performance_stream = iter_performance_rows(
        archive_path,
        sample.loan_id_set,
        stop_after_selected=True,
        validate_loan_order=True,
    )
    with performance_stream:
        panel_diagnostics = write_monthly_panel(
            performance_stream,
            origination_lookup,
            panel_path,
            vintage=selected_vintage,
            batch_rows=batch_rows,
        )
    scan_diagnostics = asdict(performance_stream.diagnostics)
    if not performance_stream.diagnostics.selection_complete:
        panel_path.unlink(missing_ok=True)
        raise RuntimeError("Not every sampled loan was found in the performance file")
    if panel_diagnostics.loan_ids != set(sample.loan_ids):
        panel_path.unlink(missing_ok=True)
        missing = sorted(set(sample.loan_ids) - panel_diagnostics.loan_ids)[:10]
        raise RuntimeError(f"Performance panel is missing sampled loans: {missing}")

    transition_panel = pd.read_parquet(
        panel_path, columns=["loan_id", "reporting_month", "state"]
    )
    transition_result = estimate_transitions(transition_panel)
    transition_outputs = save_transition_outputs(
        transition_result.counts,
        transition_result.probabilities,
        transition_result.diagnostics.to_dict(),
        table_directory=table_directory,
        figure_directory=figure_directory,
        prefix=prefix,
    )
    state_distribution = _state_distribution(panel_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_distribution.to_csv(state_path, index=False, float_format="%.10f")

    panel_payload = panel_diagnostics.to_dict()
    panel_payload.pop("loan_ids")
    summary: dict[str, Any] = {
        "run_completed_at_utc": utc_timestamp(),
        "release": RELEASE_NUMBER,
        "vintage": selected_vintage,
        "sample_method": "deterministic origination file prefix",
        "sample_is_representative": False,
        "requested_loans": loan_limit,
        "archive": archive_audit,
        "performance_scan": scan_diagnostics,
        "panel": panel_payload,
        "transitions": transition_result.diagnostics.to_dict(),
        "actual_loss_validation_passed": (
            panel_diagnostics.actual_loss_rows > 0
            and panel_diagnostics.actual_loss_outside_tolerance_rows == 0
        ),
        "outputs": {
            "origination_sample": str(origination_path),
            "monthly_panel": str(panel_path),
            "state_distribution": str(state_path),
            "archive_audit": str(audit_path),
            **transition_outputs,
        },
    }
    write_json(summary, summary_path)
    if panel_diagnostics.actual_loss_rows == 0:
        raise RuntimeError("Phase 0 sample contains no disclosed Actual Loss rows")
    if panel_diagnostics.actual_loss_outside_tolerance_rows:
        raise RuntimeError(
            "One or more disclosed Actual Loss rows failed Release 47 reconciliation"
        )
    return summary


def summary_as_text(summary: dict[str, Any]) -> str:
    """Return a stable command line representation of the completed run."""
    return json.dumps(summary, indent=2, sort_keys=True, default=str)


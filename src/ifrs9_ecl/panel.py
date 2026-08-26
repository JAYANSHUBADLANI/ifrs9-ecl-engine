"""Streaming construction of a Release 47 monthly loan panel."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from ifrs9_ecl.losses import reconcile_actual_loss
from ifrs9_ecl.schemas import ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS
from ifrs9_ecl.states import derive_state

ORIGINATION_PANEL_COLUMNS = tuple(
    f"orig_{column}" for column in ORIGINATION_COLUMNS if column != "loan_id"
)
PANEL_COLUMNS = (
    "vintage",
    "state",
    "is_pre_may_2019_cycle",
    "is_before_first_payment_cycle",
    *PERFORMANCE_COLUMNS,
    *ORIGINATION_PANEL_COLUMNS,
)
PANEL_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            pa.bool_()
            if column in {"is_pre_may_2019_cycle", "is_before_first_payment_cycle"}
            else pa.string(),
        )
        for column in PANEL_COLUMNS
    ]
)


@dataclass
class PanelBuildDiagnostics:
    """Auditable counts collected while the monthly panel is written."""

    vintage: str
    rows_written: int = 0
    loan_ids: set[str] = field(default_factory=set)
    state_counts: Counter[str] = field(default_factory=Counter)
    zero_balance_code_counts: Counter[str] = field(default_factory=Counter)
    earliest_reporting_month: str | None = None
    latest_reporting_month: str | None = None
    pre_may_2019_rows: int = 0
    may_2019_or_later_rows: int = 0
    before_first_payment_rows: int = 0
    actual_loss_rows: int = 0
    actual_loss_reconciled_rows: int = 0
    actual_loss_outside_tolerance_rows: int = 0
    maximum_actual_loss_difference: Decimal = Decimal("0")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["loan_ids"] = sorted(self.loan_ids)
        payload["loan_count"] = len(self.loan_ids)
        payload["state_counts"] = dict(sorted(self.state_counts.items()))
        payload["zero_balance_code_counts"] = dict(
            sorted(self.zero_balance_code_counts.items())
        )
        payload["maximum_actual_loss_difference"] = str(
            self.maximum_actual_loss_difference
        )
        return payload


def _is_before_first_payment(reporting_month: str, first_payment_date: str) -> bool:
    return (
        len(reporting_month) == 6
        and reporting_month.isdigit()
        and len(first_payment_date) == 6
        and first_payment_date.isdigit()
        and reporting_month < first_payment_date
    )


def build_panel_record(
    performance: Mapping[str, str],
    origination: Mapping[str, str],
    vintage: str,
) -> dict[str, object]:
    """Join one performance row to its origination attributes and derive state."""
    if performance.get("loan_id") != origination.get("loan_id"):
        raise ValueError("Performance and origination loan IDs do not match")
    reporting_month = performance.get("reporting_month", "")
    delinquency_status = performance.get("delinquency_status", "")
    first_payment_date = origination.get("first_payment_date", "")
    before_first_payment = _is_before_first_payment(
        reporting_month, first_payment_date
    )
    row: dict[str, object] = {
        "vintage": vintage,
        "state": derive_state(
            delinquency_status, performance.get("zero_balance_code", "")
        ),
        "is_pre_may_2019_cycle": bool(
            len(reporting_month) == 6
            and reporting_month.isdigit()
            and reporting_month <= "201904"
        ),
        "is_before_first_payment_cycle": bool(
            before_first_payment and delinquency_status.strip() in {"0", "00"}
        ),
    }
    row.update({column: performance.get(column, "") for column in PERFORMANCE_COLUMNS})
    row.update(
        {
            f"orig_{column}": origination.get(column, "")
            for column in ORIGINATION_COLUMNS
            if column != "loan_id"
        }
    )
    return row


def _write_batch(writer: pq.ParquetWriter, batch: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(batch, schema=PANEL_SCHEMA)
    writer.write_table(table)


def write_monthly_panel(
    performance_rows: Iterable[Mapping[str, str]],
    origination_lookup: Mapping[str, Mapping[str, str]],
    output_path: Path,
    *,
    vintage: str,
    batch_rows: int = 50_000,
) -> PanelBuildDiagnostics:
    """Stream selected performance rows into a compressed, joined Parquet panel."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    diagnostics = PanelBuildDiagnostics(vintage=vintage)
    writer = pq.ParquetWriter(
        temporary_path,
        PANEL_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    batch: list[dict[str, object]] = []
    try:
        for performance in performance_rows:
            loan_id = performance.get("loan_id", "")
            if not loan_id:
                raise ValueError("Performance row has a blank loan ID")
            try:
                origination = origination_lookup[loan_id]
            except KeyError as exc:
                raise KeyError(
                    f"No sampled origination record for performance loan {loan_id}"
                ) from exc
            row = build_panel_record(performance, origination, vintage)
            batch.append(row)
            diagnostics.rows_written += 1
            diagnostics.loan_ids.add(loan_id)
            diagnostics.state_counts[str(row["state"])] += 1
            zero_balance_code = str(performance.get("zero_balance_code", "")).strip()
            if zero_balance_code:
                diagnostics.zero_balance_code_counts[zero_balance_code] += 1
            reporting_month = str(performance.get("reporting_month", ""))
            if reporting_month:
                diagnostics.earliest_reporting_month = min(
                    diagnostics.earliest_reporting_month or reporting_month,
                    reporting_month,
                )
                diagnostics.latest_reporting_month = max(
                    diagnostics.latest_reporting_month or reporting_month,
                    reporting_month,
                )
            if bool(row["is_pre_may_2019_cycle"]):
                diagnostics.pre_may_2019_rows += 1
            else:
                diagnostics.may_2019_or_later_rows += 1
            if bool(row["is_before_first_payment_cycle"]):
                diagnostics.before_first_payment_rows += 1

            reconciliation = reconcile_actual_loss(performance)
            if reconciliation.actual_loss is not None:
                diagnostics.actual_loss_rows += 1
                if reconciliation.is_within_tolerance:
                    diagnostics.actual_loss_reconciled_rows += 1
                else:
                    diagnostics.actual_loss_outside_tolerance_rows += 1
                diagnostics.maximum_actual_loss_difference = max(
                    diagnostics.maximum_actual_loss_difference,
                    reconciliation.absolute_difference or Decimal("0"),
                )
            if len(batch) >= batch_rows:
                _write_batch(writer, batch)
                batch.clear()
        if batch:
            _write_batch(writer, batch)
            batch.clear()
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    writer.close()
    if diagnostics.rows_written == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("No selected performance rows were written")
    temporary_path.replace(output_path)
    return diagnostics


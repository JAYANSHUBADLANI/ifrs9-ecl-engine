"""Phase 5 provision backtesting against later disclosed Actual Loss.

The reporting-date provision is fixed before any later loss record is read.
Loss events are filtered strictly after the snapshot month, discounted event by
event, and then aggregated to one observation per loan before backtesting. This
avoids duplicating a loan's provision when more than one loss row is disclosed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .backtest import BacktestResult, backtest_provisions, discount_realized_losses
from .config import ProjectConfig
from .utils import month_number, utc_timestamp, write_json


PHASE5_ARTIFACT_VERSION = 3
DEFAULT_PHASE4_PATH = Path("phase4/loan_ecl.parquet")
DEFAULT_REALIZED_LOSS_PATH = Path("phase3/realized_losses.parquet")


@dataclass(frozen=True)
class Phase5Calculation:
    """Prepared loss events and provision backtest outputs."""

    result: BacktestResult
    realized_loss_events: pd.DataFrame
    observations: pd.DataFrame


def _month_text(value: Any, name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{name} must be nonmissing")
    if isinstance(value, (int, np.integer)):
        text = f"{int(value):06d}"
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be YYYYMM")
        text = f"{int(value):06d}"
    else:
        text = str(value).strip().replace("-", "")
    month_number(text)
    return text


def _month_ordinals(values: pd.Series, name: str) -> np.ndarray:
    return np.asarray(
        [month_number(_month_text(value, name)) for value in values],
        dtype="int64",
    )


def _segment_columns(
    observations: pd.DataFrame,
    segment_columns: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if segment_columns is None:
        selected = ("stage",) if "stage" in observations.columns else ()
    elif isinstance(segment_columns, str):
        selected = (segment_columns,)
    else:
        selected = tuple(segment_columns)
    if len(set(selected)) != len(selected):
        raise ValueError("segment_columns contains duplicates")
    missing = [column for column in selected if column not in observations.columns]
    if missing:
        raise KeyError(f"backtest observations are missing segment columns: {missing}")
    return selected


def prepare_realized_loss_events(
    losses: pd.DataFrame,
    loans: pd.DataFrame,
    *,
    snapshot_month: Any,
    cutoff_month: Any,
    loan_id_col: str = "loan_id",
    loss_col: str = "actual_loss",
    loss_month_col: str = "reporting_month",
    effective_rate_col: str = "effective_annual_rate",
    align_to_ecl_horizon: bool = True,
) -> pd.DataFrame:
    """Filter, discount, and retain later Actual Loss events for snapshot loans.

    The primary accuracy backtest aligns realized losses with the provision
    horizon: 12 months for Stage 1, the modeled remaining lifetime for Stage 2,
    and the full data window for Stage 3 resolution. Setting
    ``align_to_ecl_horizon`` to false supports a separate ultimate-loss stress
    diagnostic, but that result is not directly comparable with Stage 1 ECL.
    """
    if not isinstance(align_to_ecl_horizon, (bool, np.bool_)):
        raise TypeError("align_to_ecl_horizon must be boolean")
    required_loans = [loan_id_col]
    if bool(align_to_ecl_horizon):
        required_loans.extend(["stage", "ecl_horizon_months"])
    missing_loans = [column for column in required_loans if column not in loans.columns]
    if missing_loans:
        raise KeyError(f"loan table is missing columns: {missing_loans}")
    if loans[loan_id_col].isna().any() or loans[loan_id_col].duplicated().any():
        raise ValueError("loan table IDs must be nonmissing and unique")

    loss_aliases = [loss_col, "realized_loss", "disclosed_actual_loss"]
    month_aliases = [loss_month_col, "realized_loss_month", "loss_month"]
    selected_loss_col = next(
        (column for column in loss_aliases if column in losses), None
    )
    selected_month_col = next(
        (column for column in month_aliases if column in losses), None
    )
    if loan_id_col not in losses.columns:
        raise KeyError(f"loss table is missing {loan_id_col!r}")
    if selected_loss_col is None:
        raise KeyError(f"loss table is missing one of {loss_aliases}")
    if selected_month_col is None:
        raise KeyError(f"loss table is missing one of {month_aliases}")

    snapshot_text = _month_text(snapshot_month, "snapshot_month")
    cutoff_text = _month_text(cutoff_month, "cutoff_month")
    snapshot_ordinal = month_number(snapshot_text)
    cutoff_ordinal = month_number(cutoff_text)
    if cutoff_ordinal <= snapshot_ordinal:
        raise ValueError("cutoff_month must be after snapshot_month")

    metadata_columns = [loan_id_col]
    if effective_rate_col in loans.columns:
        metadata_columns.append(effective_rate_col)
    if bool(align_to_ecl_horizon):
        metadata_columns.extend(["stage", "ecl_horizon_months"])
    metadata = loans[metadata_columns].copy()
    metadata = metadata.rename(columns={effective_rate_col: "effective_annual_rate"})

    events = losses[[loan_id_col, selected_month_col, selected_loss_col]].copy()
    events = events.rename(
        columns={
            selected_month_col: "realized_loss_month",
            selected_loss_col: "actual_loss",
        }
    )
    events = events[events[loan_id_col].isin(set(loans[loan_id_col]))].copy()
    events["actual_loss"] = pd.to_numeric(events["actual_loss"], errors="coerce")
    events = events[events["actual_loss"].notna()].copy()
    events = events.merge(metadata, on=loan_id_col, how="left", validate="many_to_one")
    event_columns = [
        loan_id_col,
        "realized_loss_month",
        "actual_loss",
        "months_to_loss",
        "effective_annual_rate",
        "discount_factor",
        "discounted_actual_loss",
    ]
    if bool(align_to_ecl_horizon):
        event_columns.extend(["stage", "backtest_horizon_months"])
    if events.empty:
        return pd.DataFrame(columns=event_columns)

    ordinals = _month_ordinals(events["realized_loss_month"], "realized_loss_month")
    in_window = (ordinals > snapshot_ordinal) & (ordinals <= cutoff_ordinal)
    if bool(align_to_ecl_horizon):
        stages = pd.to_numeric(events["stage"], errors="raise")
        if stages.isna().any() or not stages.isin([1, 2, 3]).all():
            raise ValueError("stage must contain only 1, 2, or 3")
        modeled_horizons = pd.to_numeric(events["ecl_horizon_months"], errors="raise")
        finite_horizons = np.isfinite(modeled_horizons)
        performing = stages != 3
        if (
            modeled_horizons.isna().any()
            or not finite_horizons.all()
            or (modeled_horizons[performing] <= 0.0).any()
        ):
            raise ValueError(
                "ecl_horizon_months must be finite and positive for Stages 1 and 2"
            )
        maximum_observation_months = cutoff_ordinal - snapshot_ordinal
        aligned_horizons = np.where(
            stages.to_numpy(dtype="int64") == 3,
            maximum_observation_months,
            np.minimum(
                modeled_horizons.to_numpy(dtype="float64"),
                maximum_observation_months,
            ),
        )
        events["backtest_horizon_months"] = aligned_horizons.astype("float64")
        in_window &= (ordinals - snapshot_ordinal) <= aligned_horizons
    events = events.loc[in_window].copy()
    ordinals = ordinals[in_window]
    events["realized_loss_month"] = [
        _month_text(value, "realized_loss_month")
        for value in events["realized_loss_month"]
    ]
    events["months_to_loss"] = (ordinals - snapshot_ordinal).astype("float64")

    if "effective_annual_rate" not in events.columns:
        events["effective_annual_rate"] = 0.0
    events["effective_annual_rate"] = pd.to_numeric(
        events["effective_annual_rate"], errors="raise"
    )
    events["discounted_actual_loss"] = discount_realized_losses(
        events["actual_loss"],
        events["months_to_loss"],
        events["effective_annual_rate"],
    )
    events["discount_factor"] = np.divide(
        events["discounted_actual_loss"],
        events["actual_loss"],
        out=np.power(
            1.0 + events["effective_annual_rate"].to_numpy(dtype="float64"),
            -events["months_to_loss"].to_numpy(dtype="float64") / 12.0,
        ),
        where=events["actual_loss"].to_numpy(dtype="float64") != 0.0,
    )
    return events.sort_values(
        [loan_id_col, "realized_loss_month"], kind="stable"
    ).reset_index(drop=True)


def build_backtest_observations(
    loans: pd.DataFrame,
    realized_loss_events: pd.DataFrame,
    *,
    provision_col: str = "probability_weighted_ecl",
    loan_id_col: str = "loan_id",
    apply_exposure_weights: bool = True,
) -> pd.DataFrame:
    """Aggregate discounted events and attach them to one provision per loan."""
    required = [loan_id_col, provision_col]
    missing = [column for column in required if column not in loans.columns]
    if missing:
        raise KeyError(f"Phase 4 loan results are missing columns: {missing}")
    if loans[loan_id_col].isna().any() or loans[loan_id_col].duplicated().any():
        raise ValueError("Phase 4 loan IDs must be nonmissing and unique")
    provision = pd.to_numeric(loans[provision_col], errors="raise")
    if provision.isna().any() or not np.isfinite(provision).all():
        raise ValueError("provisions must be finite")

    if not isinstance(apply_exposure_weights, (bool, np.bool_)):
        raise TypeError("apply_exposure_weights must be boolean")
    output = loans.copy(deep=True)
    if bool(apply_exposure_weights) and "exposure_weight" in output.columns:
        backtest_weight = pd.to_numeric(output["exposure_weight"], errors="raise")
        if (
            backtest_weight.isna().any()
            or not np.isfinite(backtest_weight).all()
            or (backtest_weight < 0.0).any()
        ):
            raise ValueError("exposure_weight must be finite and nonnegative")
    else:
        backtest_weight = pd.Series(1.0, index=output.index)
    output["backtest_weight"] = backtest_weight.astype("float64")
    output["provision_unweighted"] = provision.astype("float64")
    output["provision"] = output["provision_unweighted"] * output["backtest_weight"]
    if realized_loss_events.empty:
        output["loss_event_count"] = 0
        output["realized_loss_unweighted"] = 0.0
        output["discounted_realized_loss_unweighted"] = 0.0
        output["first_realized_loss_month"] = pd.NA
    else:
        required_event = [loan_id_col, "actual_loss", "discounted_actual_loss"]
        missing_event = [
            column for column in required_event if column not in realized_loss_events
        ]
        if missing_event:
            raise KeyError(f"realized loss events are missing columns: {missing_event}")
        grouped = (
            realized_loss_events.groupby(loan_id_col, sort=False, dropna=False)
            .agg(
                loss_event_count=("actual_loss", "size"),
                realized_loss_unweighted=("actual_loss", "sum"),
                discounted_realized_loss_unweighted=("discounted_actual_loss", "sum"),
                first_realized_loss_month=("realized_loss_month", "min"),
            )
            .reset_index()
        )
        output = output.merge(
            grouped, on=loan_id_col, how="left", validate="one_to_one"
        )
        output["loss_event_count"] = (
            output["loss_event_count"].fillna(0).astype("int64")
        )
    for column in ("realized_loss_unweighted", "discounted_realized_loss_unweighted"):
        output[column] = output[column].fillna(0.0).astype("float64")
    output["realized_loss"] = (
        output["realized_loss_unweighted"] * output["backtest_weight"]
    )
    output["discounted_realized_loss_input"] = (
        output["discounted_realized_loss_unweighted"] * output["backtest_weight"]
    )
    return output


def calculate_phase5(
    loans: pd.DataFrame,
    losses: pd.DataFrame,
    *,
    snapshot_month: Any,
    cutoff_month: Any,
    segment_columns: str | Sequence[str] | None = None,
    provision_col: str = "probability_weighted_ecl",
    loan_id_col: str = "loan_id",
    apply_exposure_weights: bool = True,
    align_to_ecl_horizon: bool = True,
) -> Phase5Calculation:
    """Backtest snapshot ECL against strictly subsequent Actual Loss."""
    events = prepare_realized_loss_events(
        losses,
        loans,
        snapshot_month=snapshot_month,
        cutoff_month=cutoff_month,
        loan_id_col=loan_id_col,
        align_to_ecl_horizon=align_to_ecl_horizon,
    )
    observations = build_backtest_observations(
        loans,
        events,
        provision_col=provision_col,
        loan_id_col=loan_id_col,
        apply_exposure_weights=apply_exposure_weights,
    )
    segments = _segment_columns(observations, segment_columns)
    result = backtest_provisions(
        observations,
        provision_col="provision",
        realized_loss_col="realized_loss",
        discounted_loss_col="discounted_realized_loss_input",
        segment_cols=segments,
    )
    return Phase5Calculation(
        result=result,
        realized_loss_events=events,
        observations=observations,
    )


def _resolve_processed_path(config: ProjectConfig, value: Any, default: Path) -> Path:
    path = Path(str(value)) if value is not None else default
    if path.is_absolute():
        return path.resolve()
    return (config.path("processed_data") / path).resolve()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _read_phase1_actual_loss_events(
    paths: Sequence[Path],
    *,
    selected_loan_ids: set[Any],
    batch_rows: int = 100_000,
) -> pd.DataFrame:
    """Scan only three needed columns from Phase 1 panels in bounded batches."""
    selected: list[pd.DataFrame] = []
    columns = ["loan_id", "reporting_month", "actual_loss"]
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=columns):
            frame = batch.to_pandas()
            mask = (
                frame["loan_id"].isin(selected_loan_ids) & frame["actual_loss"].notna()
            )
            if mask.any():
                selected.append(frame.loc[mask, columns].copy())
    if not selected:
        return pd.DataFrame(columns=columns)
    return pd.concat(selected, ignore_index=True)


def _input_signature(paths: Iterable[Path], settings: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted((path for path in paths if path.is_file()), key=str):
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    frame.to_csv(temporary, index=False, float_format="%.10f")
    temporary.replace(path)


def _backtest_figure(result: BacktestResult, path: Path) -> None:
    if result.by_segment.empty:
        labels = ["Portfolio"]
        provisions = [result.overall.total_provision]
        losses = [result.overall.total_discounted_realized_loss]
    else:
        labels = [
            " | ".join(str(row[column]) for column in result.segment_columns)
            for _, row in result.by_segment.iterrows()
        ]
        provisions = result.by_segment["total_provision"].to_numpy(dtype="float64")
        losses = result.by_segment["total_discounted_realized_loss"].to_numpy(
            dtype="float64"
        )
        coverage = result.by_segment["coverage_ratio"].to_numpy(dtype="float64")
    if result.by_segment.empty:
        coverage = np.asarray([result.overall.coverage_ratio], dtype="float64")
    x = np.arange(len(labels), dtype="float64")
    width = 0.38
    figure_width = max(10.0, 1.2 * len(labels) + 7.0)
    figure, (axis, coverage_axis) = plt.subplots(
        1,
        2,
        figsize=(figure_width, 4.8),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    axis.bar(x - width / 2, provisions, width, label="Snapshot ECL", color="#2463a6")
    axis.bar(
        x + width / 2, losses, width, label="Discounted Actual Loss", color="#d8782d"
    )
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Amount at snapshot date")
    axis.set_title("Horizon-aligned amounts")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)

    finite_coverage = np.where(np.isfinite(coverage), coverage * 100.0, np.nan)
    coverage_colors = [
        "#3d8b5a" if value >= 100.0 else "#b94a48" for value in finite_coverage
    ]
    coverage_axis.bar(x, finite_coverage, color=coverage_colors, width=0.58)
    coverage_axis.axhline(100.0, color="#333333", linewidth=1.2, linestyle="--")
    coverage_axis.set_xticks(x, labels, rotation=35, ha="right")
    coverage_axis.set_ylabel("Coverage ratio (%)")
    coverage_axis.set_title("Coverage by segment")
    coverage_axis.grid(axis="y", alpha=0.2)
    for position, value in zip(x, finite_coverage, strict=True):
        if np.isfinite(value):
            coverage_axis.text(
                position,
                value,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Provision backtest at the December 2012 snapshot")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_phase5(config: ProjectConfig) -> dict[str, Any]:
    """Run the out-of-time provision backtest and persist its checkpoints."""
    if not isinstance(config, ProjectConfig):
        raise TypeError("config must be a ProjectConfig")
    config.ensure_output_directories()
    settings = config.raw.get("phase5", {})
    modeling = config.raw.get("modeling", {})
    snapshot_month = settings.get(
        "snapshot_month", modeling.get("backtest_snapshot_month", 201212)
    )
    cutoff_month = settings.get(
        "cutoff_month", modeling.get("performance_cutoff_month", 202603)
    )
    phase4_path = _resolve_processed_path(
        config, settings.get("phase4_loan_ecl_path"), DEFAULT_PHASE4_PATH
    )
    if not phase4_path.is_file():
        raise FileNotFoundError(f"Phase 4 loan ECL checkpoint not found: {phase4_path}")
    explicit_loss_path = _resolve_processed_path(
        config, settings.get("realized_losses_path"), DEFAULT_REALIZED_LOSS_PATH
    )
    phase1_paths = sorted(
        (config.path("processed_data") / "phase1").glob("panel_*.parquet")
    )
    loss_paths = [explicit_loss_path] if explicit_loss_path.is_file() else phase1_paths
    if not loss_paths:
        raise FileNotFoundError(
            "No realized loss input or Phase 1 panel checkpoints were found"
        )
    signature_settings = {
        "snapshot_month": snapshot_month,
        "cutoff_month": cutoff_month,
        "segment_columns": settings.get("segment_columns"),
        "provision_col": settings.get("provision_col", "probability_weighted_ecl"),
        "apply_exposure_weights": settings.get("apply_exposure_weights", True),
        "align_to_ecl_horizon": settings.get("align_to_ecl_horizon", True),
    }
    signature = _input_signature([phase4_path, *loss_paths], signature_settings)

    processed = config.path("processed_data") / "phase5"
    artifact = config.path("artifacts") / "phase5"
    event_path = processed / "realized_loss_events.parquet"
    detail_path = processed / "backtest_detail.parquet"
    overall_path = artifact / "tables" / "backtest_overall.csv"
    segment_path = artifact / "tables" / "backtest_by_segment.csv"
    ultimate_overall_path = artifact / "tables" / "backtest_ultimate_overall.csv"
    ultimate_segment_path = artifact / "tables" / "backtest_ultimate_by_segment.csv"
    summary_path = artifact / "phase5_summary.json"
    figure_path = config.path("figures") / "phase5_provision_backtest.png"
    checkpoint_outputs = (
        event_path,
        detail_path,
        overall_path,
        segment_path,
        ultimate_overall_path,
        ultimate_segment_path,
        figure_path,
    )
    if (
        bool(settings.get("reuse_completed", True))
        and summary_path.is_file()
        and all(path.is_file() for path in checkpoint_outputs)
    ):
        with summary_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            existing.get("artifact_version") == PHASE5_ARTIFACT_VERSION
            and existing.get("input_signature") == signature
        ):
            return existing

    loans = _read_table(phase4_path)
    if explicit_loss_path.is_file():
        losses = _read_table(explicit_loss_path)
        loss_source = str(explicit_loss_path)
    else:
        losses = _read_phase1_actual_loss_events(
            phase1_paths,
            selected_loan_ids=set(loans["loan_id"]),
            batch_rows=int(settings.get("scan_batch_rows", 100_000)),
        )
        loss_source = "Phase 1 panel checkpoints"

    calculation = calculate_phase5(
        loans,
        losses,
        snapshot_month=snapshot_month,
        cutoff_month=cutoff_month,
        segment_columns=settings.get("segment_columns"),
        provision_col=str(settings.get("provision_col", "probability_weighted_ecl")),
        apply_exposure_weights=bool(settings.get("apply_exposure_weights", True)),
        align_to_ecl_horizon=bool(settings.get("align_to_ecl_horizon", True)),
    )
    ultimate_calculation = calculate_phase5(
        loans,
        losses,
        snapshot_month=snapshot_month,
        cutoff_month=cutoff_month,
        segment_columns=settings.get("segment_columns"),
        provision_col=str(settings.get("provision_col", "probability_weighted_ecl")),
        apply_exposure_weights=bool(settings.get("apply_exposure_weights", True)),
        align_to_ecl_horizon=False,
    )
    _atomic_parquet(calculation.realized_loss_events, event_path)
    _atomic_parquet(calculation.result.detail, detail_path)
    overall_frame = pd.DataFrame([calculation.result.overall.to_dict()])
    _atomic_csv(overall_frame, overall_path)
    _atomic_csv(calculation.result.by_segment, segment_path)
    _atomic_csv(
        pd.DataFrame([ultimate_calculation.result.overall.to_dict()]),
        ultimate_overall_path,
    )
    _atomic_csv(ultimate_calculation.result.by_segment, ultimate_segment_path)
    _backtest_figure(calculation.result, figure_path)

    overall = calculation.result.overall
    summary: dict[str, Any] = {
        "artifact_version": PHASE5_ARTIFACT_VERSION,
        "run_completed_at_utc": utc_timestamp(),
        "input_signature": signature,
        "snapshot_month": _month_text(snapshot_month, "snapshot_month"),
        "performance_cutoff_month": _month_text(cutoff_month, "cutoff_month"),
        "loss_window_rule": (
            "strictly after snapshot; Stage 1 through its 12-month ECL horizon, "
            "Stage 2 through its remaining-lifetime ECL horizon, and Stage 3 "
            "through the data cutoff"
        ),
        "snapshot_loans": len(loans),
        "realized_loss_events": len(calculation.realized_loss_events),
        "loans_with_realized_loss_events": int(
            calculation.result.detail["loss_event_count"].gt(0).sum()
        ),
        "segment_columns": list(calculation.result.segment_columns),
        "exposure_weighting_applied": bool(
            settings.get("apply_exposure_weights", True)
            and "exposure_weight" in loans.columns
        ),
        "backtest_weight_sum": float(
            calculation.result.detail["backtest_weight"].sum()
        ),
        "overall": overall.to_dict(),
        "ultimate_stress_diagnostic": {
            "loss_window_rule": "strictly after snapshot and through cutoff for every stage",
            "realized_loss_events": len(ultimate_calculation.realized_loss_events),
            "loans_with_realized_loss_events": int(
                ultimate_calculation.result.detail["loss_event_count"].gt(0).sum()
            ),
            "overall": ultimate_calculation.result.overall.to_dict(),
        },
        "inputs": {
            "phase4_loan_ecl": str(phase4_path),
            "realized_loss_source": loss_source,
        },
        "outputs": {
            "realized_loss_events": str(event_path),
            "backtest_detail": str(detail_path),
            "overall_table": str(overall_path),
            "segment_table": str(segment_path),
            "ultimate_overall_table": str(ultimate_overall_path),
            "ultimate_segment_table": str(ultimate_segment_path),
            "backtest_figure": str(figure_path),
        },
    }
    write_json(summary, summary_path)
    return summary

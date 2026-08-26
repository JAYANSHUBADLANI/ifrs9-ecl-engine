"""Phase 2 roll rates, monthly default hazards, and lifetime PD curves.

This pipeline reads only the completed Phase 1 parquet checkpoints. It builds
one-month targets from adjacent records, fits a weighted discrete-time hazard
model, validates it out of time, and projects remaining-life PD curves. Current
and origination-reference curves use the same snapshot age and remaining term
so the resulting lifetime PD comparison is suitable for SICR staging.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.pd_model import DiscreteTimeHazardModel
from ifrs9_ecl.phase1 import fico_band, ltv_band
from ifrs9_ecl.reporting import save_transition_outputs
from ifrs9_ecl.states import (
    CENSORED_DEFECT,
    CENSORED_RPL,
    DEFAULTED,
    PAID_OFF,
    STATE_ORDER,
    TERMINAL_STATES,
)
from ifrs9_ecl.survival import conditional_hazards_to_pd
from ifrs9_ecl.transitions import estimate_transitions
from ifrs9_ecl.utils import utc_timestamp


PHASE2_PIPELINE_VERSION = 1

NUMERIC_FEATURES = (
    "loan_age_months",
    "current_interest_rate",
    "estimated_ltv",
    "orig_fico",
    "orig_ltv",
    "orig_dti",
    "orig_interest_rate",
)

CATEGORICAL_FEATURES = (
    "current_state",
    "vintage",
    "occupancy_status",
    "loan_purpose",
    "modification_flag",
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PANEL_COLUMNS = (
    "loan_id",
    "reporting_month",
    "state",
    "delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "current_actual_upb",
    "current_interest_rate",
    "estimated_ltv",
    "modification_flag",
    "orig_classic_fico",
    "orig_original_ltv",
    "orig_original_dti",
    "orig_original_interest_rate",
    "orig_original_loan_term",
    "orig_occupancy_status",
    "orig_loan_purpose",
)

HAZARD_ID_COLUMNS = (
    "loan_id",
    "current_month",
    "target_month",
    "target_default",
    "split",
    "selection_probability",
    "sample_weight",
)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, **kwargs)
    temporary.replace(path)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _month_values(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return validated YYYYMM integers and monotonically increasing ordinals."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("reporting_month must contain valid YYYYMM integers")
    month_values = numeric.astype("int64")
    months = month_values % 100
    years = month_values // 100
    valid = months.between(1, 12) & years.between(1900, 2200)
    if not valid.all():
        examples = month_values.loc[~valid].head(5).tolist()
        raise ValueError(f"invalid reporting months, examples: {examples}")
    ordinals = years * 12 + months - 1
    return month_values, ordinals.astype("int64")


def _numeric(
    values: pd.Series,
    *,
    unavailable: Iterable[float] = (),
    minimum: float | None = None,
) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").astype("float64")
    unavailable_values = tuple(unavailable)
    if unavailable_values:
        result = result.mask(result.isin(unavailable_values))
    if minimum is not None:
        result = result.mask(result < minimum)
    return result


def _categorical(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    result = result.mask(result.eq(""))
    return result.fillna("Unknown").astype("string")


def prepare_adjacent_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """Clean loan-month records and attach the next adjacent monthly state.

    Exact duplicates are collapsed and conflicting loan-month states are
    excluded. All feature values in the returned row come from month t. Only
    the target state and target month are taken from month t plus one.
    """
    required = {"loan_id", "reporting_month", "state"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(f"missing adjacent-pair columns: {missing}")
    if panel["loan_id"].isna().any():
        raise ValueError("loan_id must not be missing")

    work = panel.copy()
    work["reporting_month"], work["_month_ordinal"] = _month_values(
        work["reporting_month"]
    )
    work["state"] = _categorical(work["state"]).str.upper()
    work["_row_order"] = np.arange(len(work), dtype="int64")

    keys = ["loan_id", "_month_ordinal"]
    state_counts = work.groupby(keys, sort=False, observed=True)["state"].nunique()
    conflicts = state_counts[state_counts > 1]
    if len(conflicts):
        conflict_index = pd.MultiIndex.from_tuples(conflicts.index.tolist())
        row_index = pd.MultiIndex.from_frame(work[keys])
        work = work.loc[~row_index.isin(conflict_index)].copy()

    work = (
        work.sort_values(["loan_id", "_month_ordinal", "_row_order"], kind="stable")
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )
    raw_age = _numeric(work.get("loan_age", pd.Series(np.nan, index=work.index)), minimum=0)
    age_offset = raw_age - work["_month_ordinal"]
    loan_offset = age_offset.groupby(work["loan_id"], observed=True).transform("median")
    first_month = work.groupby("loan_id", sort=False, observed=True)[
        "_month_ordinal"
    ].transform("min")
    derived_age = work["_month_ordinal"] + loan_offset
    fallback_age = work["_month_ordinal"] - first_month
    work["_loan_age_months"] = (
        raw_age.fillna(derived_age).fillna(fallback_age).clip(lower=0.0)
    )

    grouped = work.groupby("loan_id", sort=False, observed=True)
    work["next_state"] = grouped["state"].shift(-1)
    work["target_month"] = grouped["reporting_month"].shift(-1)
    work["_next_ordinal"] = grouped["_month_ordinal"].shift(-1)
    adjacent = work[
        work["next_state"].notna()
        & ((work["_next_ordinal"] - work["_month_ordinal"]) == 1)
    ].copy()
    adjacent["target_month"] = adjacent["target_month"].astype("int64")
    adjacent["from_state"] = adjacent["state"]
    adjacent["to_state"] = adjacent["next_state"]
    return adjacent.reset_index(drop=True)


def _feature_frame(rows: pd.DataFrame, vintage: str) -> pd.DataFrame:
    index = rows.index
    orig_ltv = _numeric(
        rows.get("orig_original_ltv", pd.Series(np.nan, index=index)),
        unavailable=(999,),
        minimum=0,
    )
    estimated_ltv = _numeric(
        rows.get("estimated_ltv", pd.Series(np.nan, index=index)),
        unavailable=(999,),
        minimum=0,
    ).fillna(orig_ltv)
    features = pd.DataFrame(index=index)
    features["loan_age_months"] = _numeric(
        rows.get("_loan_age_months", rows.get("loan_age", pd.Series(np.nan, index=index))),
        minimum=0,
    )
    features["current_interest_rate"] = _numeric(
        rows.get("current_interest_rate", pd.Series(np.nan, index=index)),
        minimum=0,
    )
    features["estimated_ltv"] = estimated_ltv
    features["orig_fico"] = _numeric(
        rows.get("orig_classic_fico", pd.Series(np.nan, index=index)),
        unavailable=(9999,),
        minimum=0,
    )
    features["orig_ltv"] = orig_ltv
    features["orig_dti"] = _numeric(
        rows.get("orig_original_dti", pd.Series(np.nan, index=index)),
        unavailable=(999,),
        minimum=0,
    )
    features["orig_interest_rate"] = _numeric(
        rows.get("orig_original_interest_rate", pd.Series(np.nan, index=index)),
        minimum=0,
    )
    features["current_state"] = _categorical(rows["state"]).str.upper()
    features["vintage"] = str(vintage).upper()
    features["occupancy_status"] = _categorical(
        rows.get("orig_occupancy_status", pd.Series(pd.NA, index=index))
    )
    features["loan_purpose"] = _categorical(
        rows.get("orig_loan_purpose", pd.Series(pd.NA, index=index))
    )
    features["modification_flag"] = _categorical(
        rows.get("modification_flag", pd.Series(pd.NA, index=index))
    )
    return features.reset_index(drop=True)


def deterministic_case_control_sample(
    frame: pd.DataFrame,
    *,
    modulus: int,
    seed: int,
    loan_id_col: str = "loan_id",
    month_col: str = "current_month",
    state_col: str = "current_state",
    target_col: str = "target_default",
) -> pd.DataFrame:
    """Sample only current non-events and return their inclusion probability.

    Event rows and non-current risk rows are always retained. A modulus of one
    disables sampling. The selector is a stable hash of seed, loan ID, and
    month, making results independent of input order.
    """
    if isinstance(modulus, bool) or int(modulus) != modulus or modulus < 1:
        raise ValueError("modulus must be a positive integer")
    required = {loan_id_col, month_col, state_col, target_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing case-control columns: {missing}")

    result = frame.copy()
    result["selection_probability"] = 1.0
    if modulus == 1 or result.empty:
        return result.reset_index(drop=True)

    target = pd.to_numeric(result[target_col], errors="raise").astype("int8")
    eligible = result[state_col].eq("CURRENT") & target.eq(0)
    keys = (
        str(int(seed))
        + "|"
        + result.loc[eligible, loan_id_col].astype("string")
        + "|"
        + result.loc[eligible, month_col].astype("string")
    )
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype="uint64")
    selected = (hashes % np.uint64(modulus)) == 0
    keep = ~eligible
    keep.loc[eligible] = selected
    result.loc[eligible & keep, "selection_probability"] = 1.0 / modulus
    return result.loc[keep].reset_index(drop=True)


def build_hazard_rows(
    pairs: pd.DataFrame,
    *,
    vintage: str,
    expansion_weight: float,
    development_end_month: int,
    validation_end_month: int,
    current_non_event_modulus: int,
    sample_seed: int,
) -> pd.DataFrame:
    """Build weighted development and validation rows from adjacent pairs."""
    if validation_end_month <= development_end_month:
        raise ValueError("validation_end_month must follow development_end_month")
    risk = pairs.loc[~pairs["state"].isin(TERMINAL_STATES)].copy()
    risk = risk.loc[risk["target_month"] <= validation_end_month].copy()
    if risk.empty:
        columns = list(HAZARD_ID_COLUMNS) + list(MODEL_FEATURES)
        return pd.DataFrame(columns=columns)

    features = _feature_frame(risk, vintage)
    rows = pd.concat(
        [
            risk[["loan_id"]].reset_index(drop=True),
            pd.DataFrame(
                {
                    "current_month": risk["reporting_month"].to_numpy(dtype="int64"),
                    "target_month": risk["target_month"].to_numpy(dtype="int64"),
                    "target_default": risk["next_state"].eq(DEFAULTED).to_numpy(dtype="int8"),
                }
            ),
            features,
        ],
        axis=1,
    )
    development = rows["target_month"] <= development_end_month
    validation = rows["target_month"].between(
        development_end_month + 1, validation_end_month
    )
    rows = rows.loc[development | validation].copy()
    rows["split"] = np.where(
        rows["target_month"] <= development_end_month,
        "development",
        "validation",
    )

    development_rows = deterministic_case_control_sample(
        rows.loc[rows["split"].eq("development")],
        modulus=current_non_event_modulus,
        seed=sample_seed,
    )
    validation_rows = rows.loc[rows["split"].eq("validation")].copy()
    validation_rows["selection_probability"] = 1.0
    sampled = pd.concat([development_rows, validation_rows], ignore_index=True)
    sampled["sample_weight"] = (
        float(expansion_weight) / sampled["selection_probability"]
    )
    sampled["target_default"] = sampled["target_default"].astype("int8")
    return sampled.loc[:, list(HAZARD_ID_COLUMNS) + list(MODEL_FEATURES)]


def _segment_roll_rows(
    pairs: pd.DataFrame,
    *,
    vintage: str,
    expansion_weight: float,
) -> pd.DataFrame:
    working = pairs.copy()
    working["fico_band"] = fico_band(working["orig_classic_fico"])
    working["ltv_band"] = ltv_band(working["orig_original_ltv"])
    specifications = (
        ("Vintage", pd.Series(str(vintage).upper(), index=working.index)),
        ("Classic FICO", working["fico_band"]),
        ("Original LTV", working["ltv_band"]),
    )
    outputs: list[pd.DataFrame] = []
    for segment_type, segment in specifications:
        work = working.assign(segment=segment)
        grouped = (
            work.groupby(
                ["segment", "from_state", "to_state"],
                observed=True,
                dropna=False,
            )
            .size()
            .rename("sample_count")
            .reset_index()
        )
        grouped.insert(0, "segment_type", segment_type)
        grouped["weighted_count"] = grouped["sample_count"] * float(
            expansion_weight
        )
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def _snapshot_rows(
    cleaned_panel: pd.DataFrame,
    *,
    vintage: str,
    snapshot_month: int,
    expansion_weight: float,
) -> pd.DataFrame:
    current = cleaned_panel.loc[
        cleaned_panel["reporting_month"].eq(snapshot_month)
        & ~cleaned_panel["state"].isin({PAID_OFF, CENSORED_RPL, CENSORED_DEFECT})
    ].copy()
    if current.empty:
        return pd.DataFrame()
    current = current.sort_values(["loan_id", "_month_ordinal"], kind="stable")
    if current["loan_id"].duplicated().any():
        raise ValueError(f"{vintage} has duplicate loans at the snapshot month")

    loan_ids = set(current["loan_id"])
    origination = (
        cleaned_panel.loc[cleaned_panel["loan_id"].isin(loan_ids)]
        .sort_values(["loan_id", "_month_ordinal"], kind="stable")
        .groupby("loan_id", sort=False, observed=True)
        .head(1)
        .copy()
    )
    current_features = _feature_frame(current, vintage)
    origination_features = _feature_frame(origination, vintage)
    origination_features["loan_id"] = origination["loan_id"].to_numpy()
    origination_features["origination_reporting_month"] = origination[
        "reporting_month"
    ].to_numpy(dtype="int64")

    # SICR requires like-for-like remaining-life PDs. Keep the snapshot age for
    # the reference curve while resetting the remaining risk covariates.
    origination_features = origination_features.rename(
        columns={column: f"orig__{column}" for column in MODEL_FEATURES}
    )
    base = pd.concat(
        [current[["loan_id", "reporting_month", "state"]].reset_index(drop=True), current_features],
        axis=1,
    )
    base = base.merge(origination_features, on="loan_id", how="left", validate="one_to_one")
    base["orig__loan_age_months"] = base["loan_age_months"]

    remaining = _numeric(
        current["remaining_months_to_legal_maturity"], minimum=0
    ).reset_index(drop=True)
    original_term = _numeric(
        current["orig_original_loan_term"], minimum=0
    ).reset_index(drop=True)
    base["remaining_term_months"] = remaining.fillna(original_term)
    base["current_actual_upb"] = _numeric(
        current["current_actual_upb"], minimum=0
    ).reset_index(drop=True)
    current_rate = _numeric(current["current_interest_rate"], minimum=0).reset_index(drop=True)
    original_rate = _numeric(
        current["orig_original_interest_rate"], minimum=0
    ).reset_index(drop=True)
    base["effective_annual_rate"] = current_rate.fillna(original_rate) / 100.0
    base["days_past_due"] = base["state"].map(
        {
            "CURRENT": 0,
            "30_DPD": 30,
            "60_DPD": 60,
            "90_PLUS": 90,
            DEFAULTED: 90,
        }
    ).fillna(0).astype("int64")
    base["is_default"] = base["state"].eq(DEFAULTED)
    base["fico_band"] = fico_band(current["orig_classic_fico"]).reset_index(drop=True)
    base["ltv_band"] = ltv_band(current["orig_original_ltv"]).reset_index(drop=True)
    base["expansion_weight"] = float(expansion_weight)
    base["vintage"] = str(vintage).upper()
    return base


def _checkpoint_paths(config: ProjectConfig, vintage: str) -> dict[str, Path]:
    key = vintage.lower()
    processed = config.path("processed_data") / "phase2" / "quarters"
    artifacts = config.path("artifacts") / "phase2" / "quarters"
    return {
        "hazard_rows": processed / f"{key}_hazard_rows.parquet",
        "roll_rows": processed / f"{key}_roll_rows.parquet",
        "snapshot_rows": processed / f"{key}_snapshot_rows.parquet",
        "metadata": artifacts / f"{key}.json",
    }


def _phase1_descriptor(config: ProjectConfig, vintage: str) -> dict[str, Any]:
    key = vintage.lower()
    panel_path = config.path("processed_data") / "phase1" / f"panel_{key}.parquet"
    summary_path = config.path("artifacts") / "phase1" / "quarters" / f"{key}.json"
    if not panel_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 checkpoint is incomplete for {vintage}: "
            f"expected {panel_path} and {summary_path}"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    parquet = pq.ParquetFile(panel_path)
    expected_rows = summary.get("panel", {}).get("rows_written")
    if expected_rows is not None and parquet.metadata.num_rows != int(expected_rows):
        raise RuntimeError(f"Phase 1 row count does not reconcile for {vintage}")
    stat = panel_path.stat()
    return {
        "vintage": str(vintage).upper(),
        "panel_path": str(panel_path),
        "summary_path": str(summary_path),
        "panel_rows": parquet.metadata.num_rows,
        "panel_size": stat.st_size,
        "panel_mtime_ns": stat.st_mtime_ns,
        "release": summary.get("release"),
        "sample_seed": summary.get("sample_seed"),
        "sampled_loans": summary.get("sampled_loans"),
        "expansion_weight": float(summary["expansion_weight"]),
    }


def _quarter_contract(
    descriptor: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": PHASE2_PIPELINE_VERSION,
        "source": descriptor,
        "development_end_month": settings["development_end_month"],
        "validation_end_month": settings["validation_end_month"],
        "snapshot_month": settings["snapshot_month"],
        "current_non_event_modulus": settings["current_non_event_modulus"],
        "sample_seed": settings["sample_seed"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }


def _reusable_quarter(
    paths: dict[str, Path],
    *,
    contract_hash: str,
) -> dict[str, Any] | None:
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        with paths["metadata"].open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return None
    if metadata.get("contract_hash") != contract_hash:
        return None
    row_keys = {
        "hazard_rows": "hazard_rows",
        "roll_rows": "roll_rows",
        "snapshot_rows": "snapshot_rows",
    }
    for path_key, count_key in row_keys.items():
        rows = pq.ParquetFile(paths[path_key]).metadata.num_rows
        if rows != int(metadata[count_key]):
            return None
    return metadata


def build_quarter_checkpoint(
    config: ProjectConfig,
    descriptor: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Build or reuse one Phase 2 quarter checkpoint."""
    vintage = str(descriptor["vintage"])
    paths = _checkpoint_paths(config, vintage)
    contract = _quarter_contract(descriptor, settings)
    contract_hash = _fingerprint(contract)
    existing = _reusable_quarter(paths, contract_hash=contract_hash)
    if existing is not None:
        return existing

    panel = pd.read_parquet(descriptor["panel_path"], columns=list(PANEL_COLUMNS))
    pairs = prepare_adjacent_pairs(panel)
    transition = estimate_transitions(
        panel[["loan_id", "reporting_month", "state"]]
    )
    overall = transition.long[["from_state", "to_state", "count"]].rename(
        columns={"count": "sample_count"}
    )
    overall.insert(0, "segment", "All")
    overall.insert(0, "segment_type", "Overall")
    overall["weighted_count"] = (
        overall["sample_count"] * float(descriptor["expansion_weight"])
    )
    segmented = _segment_roll_rows(
        pairs,
        vintage=vintage,
        expansion_weight=float(descriptor["expansion_weight"]),
    )
    roll_rows = pd.concat([overall, segmented], ignore_index=True)
    roll_rows.insert(0, "vintage", vintage)

    hazard_rows = build_hazard_rows(
        pairs,
        vintage=vintage,
        expansion_weight=float(descriptor["expansion_weight"]),
        development_end_month=int(settings["development_end_month"]),
        validation_end_month=int(settings["validation_end_month"]),
        current_non_event_modulus=int(settings["current_non_event_modulus"]),
        sample_seed=int(settings["sample_seed"]),
    )
    snapshot_rows = _snapshot_rows(
        _cleaned_panel(panel),
        vintage=vintage,
        snapshot_month=int(settings["snapshot_month"]),
        expansion_weight=float(descriptor["expansion_weight"]),
    )

    _atomic_parquet(hazard_rows, paths["hazard_rows"])
    _atomic_parquet(roll_rows, paths["roll_rows"])
    _atomic_parquet(snapshot_rows, paths["snapshot_rows"])
    split_counts = hazard_rows["split"].value_counts().to_dict()
    metadata = {
        "pipeline_version": PHASE2_PIPELINE_VERSION,
        "vintage": vintage,
        "contract": contract,
        "contract_hash": contract_hash,
        "hazard_rows": len(hazard_rows),
        "development_rows": int(split_counts.get("development", 0)),
        "validation_rows": int(split_counts.get("validation", 0)),
        "hazard_events": int(hazard_rows["target_default"].sum()),
        "roll_rows": len(roll_rows),
        "snapshot_rows": len(snapshot_rows),
        "transition_diagnostics": transition.diagnostics.to_dict(),
        "outputs": {name: str(path) for name, path in paths.items() if name != "metadata"},
    }
    _atomic_json(metadata, paths["metadata"])
    return metadata


def _cleaned_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the loan-month cleanup used by adjacent targets and snapshots."""
    required = {"loan_id", "reporting_month", "state"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise KeyError(f"missing panel columns: {missing}")
    work = panel.copy()
    work["reporting_month"], work["_month_ordinal"] = _month_values(
        work["reporting_month"]
    )
    work["state"] = _categorical(work["state"]).str.upper()
    work["_row_order"] = np.arange(len(work), dtype="int64")
    keys = ["loan_id", "_month_ordinal"]
    state_counts = work.groupby(keys, sort=False, observed=True)["state"].nunique()
    conflicts = state_counts[state_counts > 1]
    if len(conflicts):
        conflict_index = pd.MultiIndex.from_tuples(conflicts.index.tolist())
        row_index = pd.MultiIndex.from_frame(work[keys])
        work = work.loc[~row_index.isin(conflict_index)].copy()
    work = (
        work.sort_values(["loan_id", "_month_ordinal", "_row_order"], kind="stable")
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )
    raw_age = _numeric(work.get("loan_age", pd.Series(np.nan, index=work.index)), minimum=0)
    age_offset = raw_age - work["_month_ordinal"]
    loan_offset = age_offset.groupby(work["loan_id"], observed=True).transform("median")
    first_month = work.groupby("loan_id", sort=False, observed=True)[
        "_month_ordinal"
    ].transform("min")
    work["_loan_age_months"] = (
        raw_age.fillna(work["_month_ordinal"] + loan_offset)
        .fillna(work["_month_ordinal"] - first_month)
        .clip(lower=0.0)
    )
    return work


def _aggregate_roll_rates(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(
            ["segment_type", "segment", "from_state", "to_state"],
            observed=True,
            dropna=False,
        )[["sample_count", "weighted_count"]]
        .sum()
        .reset_index()
    )
    totals = grouped.groupby(
        ["segment_type", "segment", "from_state"], observed=True
    )["weighted_count"].transform("sum")
    grouped["probability"] = grouped["weighted_count"] / totals
    return grouped.sort_values(
        ["segment_type", "segment", "from_state", "to_state"], kind="stable"
    ).reset_index(drop=True)


def weighted_validation_metrics(
    target: Sequence[int] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
    sample_weight: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Calculate discrimination and calibration metrics with exposure weights."""
    y = np.asarray(target, dtype="int8")
    p = np.asarray(predicted, dtype="float64")
    weight = np.asarray(sample_weight, dtype="float64")
    if y.ndim != 1 or p.ndim != 1 or weight.ndim != 1:
        raise ValueError("validation arrays must be one-dimensional")
    if not (len(y) == len(p) == len(weight)) or len(y) == 0:
        raise ValueError("validation arrays must have equal lengths")
    if len(y) == 0:
        raise ValueError("at least one validation row is required")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("target must contain only zero and one")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("predicted probabilities must be finite and between zero and one")
    if not np.isfinite(weight).all() or (weight < 0.0).any() or weight.sum() <= 0:
        raise ValueError("sample weights must be finite, nonnegative, and have positive total")

    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    total_weight = float(weight.sum())
    weighted_events = float(np.dot(weight, y))
    weighted_prediction = float(np.dot(weight, p) / total_weight)
    brier = float(np.dot(weight, np.square(p - y)) / total_weight)
    both_classes = len(np.unique(y[weight > 0])) == 2
    return {
        "observations": len(y),
        "events": int(y.sum()),
        "weight_sum": total_weight,
        "weighted_events": weighted_events,
        "weighted_event_rate": weighted_events / total_weight,
        "weighted_mean_prediction": weighted_prediction,
        "weighted_brier_score": brier,
        "weighted_log_loss": float(log_loss(y, clipped, sample_weight=weight, labels=[0, 1])),
        "weighted_roc_auc": (
            float(roc_auc_score(y, p, sample_weight=weight)) if both_classes else None
        ),
        "weighted_average_precision": (
            float(average_precision_score(y, p, sample_weight=weight))
            if both_classes
            else None
        ),
    }


def _calibration_table(
    validation: pd.DataFrame,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    work = validation.copy()
    quantiles = min(int(bins), len(work))
    if quantiles < 1:
        return pd.DataFrame()
    rank = work["predicted_hazard"].rank(method="first", pct=True)
    work["score_bin"] = np.minimum(
        np.ceil(rank * quantiles).astype("int64"), quantiles
    )
    work["weighted_target"] = work["sample_weight"] * work["target_default"]
    work["weighted_prediction"] = (
        work["sample_weight"] * work["predicted_hazard"]
    )
    grouped = (
        work.groupby("score_bin", observed=True)
        .agg(
            observations=("target_default", "size"),
            events=("target_default", "sum"),
            weight_sum=("sample_weight", "sum"),
            weighted_events=("weighted_target", "sum"),
            weighted_predictions=("weighted_prediction", "sum"),
            minimum_prediction=("predicted_hazard", "min"),
            maximum_prediction=("predicted_hazard", "max"),
        )
        .reset_index()
    )
    grouped["observed_default_rate"] = (
        grouped["weighted_events"] / grouped["weight_sum"]
    )
    grouped["mean_predicted_hazard"] = (
        grouped["weighted_predictions"] / grouped["weight_sum"]
    )
    return grouped


def project_lifetime_curves(
    model: DiscreteTimeHazardModel,
    features: pd.DataFrame,
    *,
    horizon_months: int,
    remaining_term_months: Sequence[float] | np.ndarray | pd.Series,
    force_default: Sequence[bool] | np.ndarray | pd.Series | None = None,
    batch_size: int = 500,
) -> tuple[list[list[float]], np.ndarray, np.ndarray]:
    """Project age-forward marginal PDs and cap them at contractual term."""
    if horizon_months < 1 or batch_size < 1:
        raise ValueError("horizon_months and batch_size must be positive")
    remaining = np.asarray(remaining_term_months, dtype="float64")
    if remaining.ndim != 1 or len(remaining) != len(features):
        raise ValueError("remaining_term_months must have one value per loan")
    remaining = np.nan_to_num(remaining, nan=float(horizon_months))
    remaining = np.clip(np.floor(remaining), 0, horizon_months).astype("int64")
    defaults = (
        np.zeros(len(features), dtype=bool)
        if force_default is None
        else np.asarray(force_default, dtype=bool)
    )
    if defaults.ndim != 1 or len(defaults) != len(features):
        raise ValueError("force_default must have one value per loan")

    marginal_lists: list[list[float]] = []
    twelve_month = np.empty(len(features), dtype="float64")
    lifetime = np.empty(len(features), dtype="float64")
    months = np.arange(1, horizon_months + 1)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        block = features.iloc[start:stop]
        hazards = model.predict_hazard_curve(block, horizon_months)
        hazards[months[None, :] > remaining[start:stop, None]] = 0.0
        block_defaults = defaults[start:stop]
        if block_defaults.any():
            hazards[block_defaults] = 0.0
            hazards[block_defaults, 0] = 1.0
        curves = conditional_hazards_to_pd(hazards, axis=1)
        marginal = curves.marginal_pd
        marginal_lists.extend(row.tolist() for row in marginal)
        twelve_month[start:stop] = marginal[:, : min(12, horizon_months)].sum(axis=1)
        lifetime[start:stop] = marginal.sum(axis=1)
    return marginal_lists, twelve_month, lifetime


def _model_coefficients(model: DiscreteTimeHazardModel) -> pd.DataFrame:
    assert model.pipeline_ is not None
    preprocessor = model.pipeline_.named_steps["preprocessor"]
    classifier = model.pipeline_.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_.reshape(-1)
    rows = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
        }
    )
    intercept = float(classifier.intercept_.reshape(-1)[0])
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "feature": ["intercept"],
                    "coefficient": [intercept],
                    "odds_ratio": [np.exp(intercept)],
                }
            ),
            rows,
        ],
        ignore_index=True,
    )


def _curve_artifacts(
    model: DiscreteTimeHazardModel,
    quarter_metadata: Sequence[dict[str, Any]],
    *,
    artifact_directory: Path,
    horizon_months: int,
    batch_size: int = 500,
) -> tuple[dict[str, str], pd.DataFrame, int]:
    current_path = artifact_directory / "current_pd_curves.parquet"
    origination_path = artifact_directory / "origination_pd_curves.parquet"
    snapshot_path = artifact_directory / "snapshot_loans.parquet"
    paths = {
        "current": current_path,
        "origination": origination_path,
        "snapshot": snapshot_path,
    }
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()
    }
    writers: dict[str, pq.ParquetWriter | None] = {
        name: None for name in paths
    }
    row_counts = {name: 0 for name in paths}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for temporary in temporary_paths.values():
        temporary.unlink(missing_ok=True)
    profile_current = np.zeros(horizon_months, dtype="float64")
    profile_origination = np.zeros(horizon_months, dtype="float64")
    profile_weight = 0.0
    seen_ids: set[str] = set()

    def write_block(name: str, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writers[name]
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_paths[name], table.schema, compression="zstd"
            )
            writers[name] = writer
        writer.write_table(table)
        row_counts[name] += len(frame)

    try:
        for metadata in quarter_metadata:
            snapshot = pd.read_parquet(metadata["outputs"]["snapshot_rows"])
            if snapshot.empty:
                continue
            identifiers = snapshot["loan_id"].astype("string")
            duplicates = set(identifiers).intersection(seen_ids)
            if duplicates:
                examples = sorted(duplicates)[:5]
                raise ValueError(
                    f"loan IDs repeat across vintages, examples: {examples}"
                )
            seen_ids.update(identifiers.tolist())
            remaining = snapshot["remaining_term_months"]
            current_lists, current_12m, current_lifetime = project_lifetime_curves(
                model,
                snapshot.loc[:, MODEL_FEATURES],
                horizon_months=horizon_months,
                remaining_term_months=remaining,
                force_default=snapshot["is_default"],
                batch_size=batch_size,
            )
            origination_features = snapshot.loc[
                :, [f"orig__{column}" for column in MODEL_FEATURES]
            ].rename(
                columns={f"orig__{column}": column for column in MODEL_FEATURES}
            )
            (
                origination_lists,
                origination_12m,
                origination_lifetime,
            ) = project_lifetime_curves(
                model,
                origination_features,
                horizon_months=horizon_months,
                remaining_term_months=remaining,
                batch_size=batch_size,
            )

            common = snapshot[["loan_id", "vintage"]].reset_index(drop=True)
            current_output = common.copy()
            current_output["curve_horizon_months"] = int(horizon_months)
            current_output["remaining_term_months"] = remaining.to_numpy(
                dtype="float64"
            )
            current_output["current_marginal_pd"] = current_lists
            current_output["current_12m_pd"] = current_12m
            current_output["current_lifetime_pd"] = current_lifetime
            origination_output = common.copy()
            origination_output["curve_horizon_months"] = int(horizon_months)
            origination_output["remaining_term_months"] = remaining.to_numpy(
                dtype="float64"
            )
            origination_output["origination_marginal_pd"] = origination_lists
            origination_output["origination_12m_pd"] = origination_12m
            origination_output["origination_lifetime_pd"] = origination_lifetime

            public_columns = [
                "loan_id",
                "vintage",
                "reporting_month",
                "state",
                "days_past_due",
                "is_default",
                "effective_annual_rate",
                "current_actual_upb",
                "loan_age_months",
                "remaining_term_months",
                "fico_band",
                "ltv_band",
                "expansion_weight",
            ]
            public = snapshot.loc[:, public_columns].copy()
            public["curve_horizon_months"] = int(horizon_months)
            public["current_12m_pd"] = current_12m
            public["current_lifetime_pd"] = current_lifetime
            public["origination_12m_pd"] = origination_12m
            public["origination_lifetime_pd"] = origination_lifetime
            write_block("current", current_output)
            write_block("origination", origination_output)
            write_block("snapshot", public)

            weights = snapshot["expansion_weight"].to_numpy(dtype="float64")
            current_matrix = np.asarray(current_lists, dtype="float64")
            origination_matrix = np.asarray(origination_lists, dtype="float64")
            profile_current += np.sum(current_matrix * weights[:, None], axis=0)
            profile_origination += np.sum(
                origination_matrix * weights[:, None], axis=0
            )
            profile_weight += float(weights.sum())

        if any(writer is None for writer in writers.values()):
            raise ValueError("no active snapshot loans were available for PD projection")
        for name, writer in writers.items():
            assert writer is not None
            writer.close()
            writers[name] = None
        for name, path in paths.items():
            temporary_paths[name].replace(path)
    except Exception:
        for name, writer in writers.items():
            if writer is not None:
                writer.close()
                writers[name] = None
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise

    mean_current_marginal = profile_current / profile_weight
    mean_origination_marginal = profile_origination / profile_weight
    profile = pd.DataFrame(
        {
            "month": np.arange(1, horizon_months + 1),
            "current_marginal_pd": mean_current_marginal,
            "current_cumulative_pd": np.cumsum(mean_current_marginal),
            "current_survival": 1.0 - np.cumsum(mean_current_marginal),
            "origination_marginal_pd": mean_origination_marginal,
            "origination_cumulative_pd": np.cumsum(mean_origination_marginal),
            "origination_survival": 1.0 - np.cumsum(mean_origination_marginal),
        }
    )
    return (
        {
            "current_pd_curves": str(current_path),
            "origination_pd_curves": str(origination_path),
            "snapshot_loans": str(snapshot_path),
        },
        profile,
        row_counts["snapshot"],
    )


def _save_figures(
    config: ProjectConfig,
    calibration: pd.DataFrame,
    profile: pd.DataFrame,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figure_directory = config.path("figures")
    figure_directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if not calibration.empty:
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        maximum = float(
            max(
                calibration["observed_default_rate"].max(),
                calibration["mean_predicted_hazard"].max(),
                1e-4,
            )
        )
        axis.plot([0, maximum], [0, maximum], color="black", linestyle=":", linewidth=1)
        axis.plot(
            calibration["mean_predicted_hazard"],
            calibration["observed_default_rate"],
            marker="o",
            color="#355f8a",
        )
        axis.set_xlabel("Weighted mean predicted monthly hazard")
        axis.set_ylabel("Weighted observed monthly default rate")
        axis.set_title("Phase 2 out-of-time calibration")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = figure_directory / "phase2_validation_calibration.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs["validation_calibration_figure"] = str(path)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        profile["month"],
        profile["current_cumulative_pd"],
        label="Current risk",
        color="#a61b1b",
    )
    axis.plot(
        profile["month"],
        profile["origination_cumulative_pd"],
        label="Origination reference",
        color="#355f8a",
    )
    axis.set_xlabel("Projection month")
    axis.set_ylabel("Weighted mean cumulative PD")
    axis.set_title("Age-forward remaining-life PD profiles")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_directory / "phase2_lifetime_pd_profile.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs["lifetime_pd_figure"] = str(path)
    return outputs


def _settings(config: ProjectConfig) -> dict[str, Any]:
    modeling = config.raw["modeling"]
    phase1 = config.raw["phase1"]
    ifrs9 = config.raw["ifrs9"]
    settings = {
        "vintages": [str(value).upper() for value in phase1["vintages"]],
        "development_end_month": int(modeling["development_end_month"]),
        "validation_end_month": int(modeling["validation_end_month"]),
        "snapshot_month": int(modeling["backtest_snapshot_month"]),
        "current_non_event_modulus": int(
            modeling.get("current_non_event_month_modulus", 1)
        ),
        "maximum_iterations": int(modeling.get("maximum_iterations", 1_000)),
        "sample_seed": int(phase1.get("sample_seed", config.raw["project"]["random_seed"])),
        "projection_months": int(ifrs9["maximum_projection_months"]),
    }
    if settings["validation_end_month"] <= settings["development_end_month"]:
        raise ValueError("validation cutoff must follow the development cutoff")
    if settings["snapshot_month"] > settings["validation_end_month"]:
        raise ValueError("snapshot month must not follow the validation cutoff")
    if settings["current_non_event_modulus"] < 1:
        raise ValueError("current non-event modulus must be positive")
    if settings["projection_months"] < 1:
        raise ValueError("maximum projection months must be positive")
    return settings


def _reusable_run(
    summary_path: Path,
    *,
    run_signature: str,
) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError):
        return None
    if summary.get("run_signature") != run_signature:
        return None
    output_values = summary.get("outputs", {}).values()
    if not output_values or not all(Path(value).is_file() for value in output_values):
        return None
    return summary


def run_phase2(config: ProjectConfig) -> dict[str, Any]:
    """Execute restart-safe Phase 2 modeling from Phase 1 checkpoints."""
    settings = _settings(config)
    config.ensure_output_directories()
    artifact_directory = config.path("artifacts") / "phase2"
    table_directory = config.path("artifacts") / "tables"
    processed_directory = config.path("processed_data") / "phase2"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    descriptors = [
        _phase1_descriptor(config, vintage) for vintage in settings["vintages"]
    ]
    run_contract = {
        "pipeline_version": PHASE2_PIPELINE_VERSION,
        "settings": settings,
        "sources": descriptors,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "reference_curve_method": (
            "snapshot age and remaining term with origination risk covariates"
        ),
    }
    run_signature = _fingerprint(run_contract)
    summary_path = config.path("artifacts") / "phase2_summary.json"
    existing = _reusable_run(summary_path, run_signature=run_signature)
    if existing is not None:
        return existing

    quarter_metadata: list[dict[str, Any]] = []
    for descriptor in descriptors:
        vintage = descriptor["vintage"]
        print(f"Phase 2 checkpoint: {vintage}", flush=True)
        quarter_metadata.append(build_quarter_checkpoint(config, descriptor, settings))

    roll_frames = [
        pd.read_parquet(metadata["outputs"]["roll_rows"])
        for metadata in quarter_metadata
    ]
    roll_rates = _aggregate_roll_rates(roll_frames)
    pooled = roll_rates.loc[
        roll_rates["segment_type"].eq("Overall") & roll_rates["segment"].eq("All")
    ]
    counts = pooled.pivot(
        index="from_state", columns="to_state", values="weighted_count"
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0.0)
    counts = counts.fillna(0.0)
    counts.index.name = "from_state"
    counts.columns.name = "to_state"
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    transition_diagnostics: dict[str, Any] = {}
    for key in quarter_metadata[0]["transition_diagnostics"]:
        values = [metadata["transition_diagnostics"][key] for metadata in quarter_metadata]
        transition_diagnostics[key] = max(values) if key == "largest_gap_months" else sum(values)
    transition_outputs = save_transition_outputs(
        counts,
        probabilities,
        transition_diagnostics,
        table_directory=table_directory,
        figure_directory=config.path("figures"),
        prefix="phase2_pooled_weighted",
    )
    pooled_path = table_directory / "phase2_pooled_roll_rates.csv"
    segmented_path = table_directory / "phase2_segmented_roll_rates.csv"
    _atomic_csv(pooled, pooled_path, float_format="%.10f")
    _atomic_csv(
        roll_rates.loc[~roll_rates["segment_type"].eq("Overall")],
        segmented_path,
        float_format="%.10f",
    )

    development_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for metadata in quarter_metadata:
        rows = pd.read_parquet(metadata["outputs"]["hazard_rows"])
        development_parts.append(rows.loc[rows["split"].eq("development")])
        validation_parts.append(rows.loc[rows["split"].eq("validation")])
    development = pd.concat(development_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    if development.empty or development["target_default"].nunique() != 2:
        raise ValueError("development data must contain default and non-default targets")
    if validation.empty:
        raise ValueError("out-of-time validation data is empty")
    development_path = processed_directory / "development_hazard_sample.parquet"
    validation_path = processed_directory / "validation_hazard_rows.parquet"
    _atomic_parquet(development, development_path)
    _atomic_parquet(validation, validation_path)

    model = DiscreteTimeHazardModel(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        age_col="loan_age_months",
        max_iter=int(settings["maximum_iterations"]),
        random_state=int(settings["sample_seed"]),
    )
    model.fit(
        development.loc[:, MODEL_FEATURES],
        development["target_default"],
        sample_weight=development["sample_weight"],
    )
    model.check_convergence(raise_on_failure=True)
    model_path = model.save(artifact_directory / "monthly_default_hazard.pkl")
    coefficients = _model_coefficients(model)
    coefficient_path = table_directory / "phase2_hazard_coefficients.csv"
    _atomic_csv(coefficients, coefficient_path, float_format="%.10f")

    validation = validation.copy()
    validation["predicted_hazard"] = model.predict_hazard(
        validation.loc[:, MODEL_FEATURES]
    )
    metrics = weighted_validation_metrics(
        validation["target_default"],
        validation["predicted_hazard"],
        validation["sample_weight"],
    )
    metrics["target_month_start"] = int(validation["target_month"].min())
    metrics["target_month_end"] = int(validation["target_month"].max())
    validation_predictions_path = processed_directory / "validation_predictions.parquet"
    prediction_columns = list(HAZARD_ID_COLUMNS) + ["vintage", "predicted_hazard"]
    _atomic_parquet(validation.loc[:, prediction_columns], validation_predictions_path)
    calibration = _calibration_table(validation)
    calibration_path = table_directory / "phase2_validation_calibration.csv"
    _atomic_csv(calibration, calibration_path, float_format="%.10f")

    curve_outputs, profile, snapshot_rows = _curve_artifacts(
        model,
        quarter_metadata,
        artifact_directory=artifact_directory,
        horizon_months=int(settings["projection_months"]),
    )
    profile_path = table_directory / "phase2_lifetime_pd_profile.csv"
    _atomic_csv(profile, profile_path, float_format="%.12f")
    figure_outputs = _save_figures(config, calibration, profile)

    development_events = int(development["target_default"].sum())
    sampled_current_non_events = development.loc[
        development["current_state"].eq("CURRENT")
        & development["target_default"].eq(0)
    ]
    summary = {
        "run_completed_at_utc": utc_timestamp(),
        "pipeline_version": PHASE2_PIPELINE_VERSION,
        "run_signature": run_signature,
        "settings": settings,
        "cutoff_contract": {
            "development_target_month_end": settings["development_end_month"],
            "validation_target_month_start": settings["development_end_month"] + 1,
            "validation_target_month_end": settings["validation_end_month"],
            "feature_timing": "all predictors are observed at month t",
            "target_timing": "default state is observed at adjacent month t plus one",
        },
        "case_control_sampling": {
            "scope": "development CURRENT non-event rows only",
            "modulus": settings["current_non_event_modulus"],
            "sampled_current_non_events": len(sampled_current_non_events),
            "inverse_probability_weighted": settings["current_non_event_modulus"] > 1,
            "events_and_delinquent_non_events_retained": True,
        },
        "development": {
            "rows": len(development),
            "events": development_events,
            "weight_sum": float(development["sample_weight"].sum()),
            "weighted_events": float(
                development.loc[development["target_default"].eq(1), "sample_weight"].sum()
            ),
        },
        "validation": metrics,
        "model_fit": model.fit_diagnostics_.to_dict(),
        "snapshot_loans": snapshot_rows,
        "lifetime_pd_reference": {
            "method": (
                "same snapshot age and remaining term, with risk covariates reset "
                "to the first observed origination baseline"
            ),
            "curve_horizon_months": settings["projection_months"],
            "contractual_term_cap_applied": True,
        },
        "quarter_checkpoints": quarter_metadata,
        "outputs": {
            **transition_outputs,
            **curve_outputs,
            **figure_outputs,
            "pooled_roll_rates": str(pooled_path),
            "segmented_roll_rates": str(segmented_path),
            "development_hazard_sample": str(development_path),
            "validation_hazard_rows": str(validation_path),
            "validation_predictions": str(validation_predictions_path),
            "validation_calibration": str(calibration_path),
            "hazard_model": str(model_path),
            "hazard_coefficients": str(coefficient_path),
            "lifetime_pd_profile": str(profile_path),
        },
    }
    serializable_summary = json.loads(json.dumps(summary, default=str))
    _atomic_json(serializable_summary, summary_path)
    return serializable_summary

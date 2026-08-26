"""Phase 3 cure, LGD, and amortizing EAD integration pipeline.

The pipeline consumes the compact Phase 1 quarter panels. Monthly panels are
opened one quarter at a time, converted to compact outcome checkpoints, and
released before the next quarter is opened. Model fitting therefore depends on
episode, disposition, validation, and snapshot rows rather than the full
monthly history in memory.

The cure target is deliberately explicit. A cure requires three consecutive
adjacent CURRENT months within a 12 month outcome horizon by default. Panels
are truncated at each split cutoff before outcomes are constructed. Code 15 is
validated as a default disposition and code 16 is validated as a censored RPL
exit. Realized LGD always retains disclosed Actual Loss divided by Zero Balance
Removal UPB before any modeling treatment.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import ProjectConfig
from .ead import calculate_default_ead, project_amortizing_ead, scheduled_balance_at_month
from .lgd import (
    CureDefinition,
    ConditionalSeverityModel,
    WeightedCureModel,
    add_realized_lgd,
    build_delinquency_episodes,
    fit_conditional_severity_model,
    fit_weighted_cure_model,
)
from .losses import reconcile_actual_loss
from .states import (
    CENSORED_RPL,
    CURRENT,
    DEFAULTED,
    DPD_30,
    DPD_60,
    DPD_90_PLUS,
)
from .utils import utc_timestamp


DEFAULT_ZERO_BALANCE_CODES = frozenset({"02", "03", "09", "15"})
RPL_CENSOR_CODE = "16"
ACTIVE_SNAPSHOT_STATES = frozenset({CURRENT, DPD_30, DPD_60, DPD_90_PLUS})
REPORTING_SNAPSHOT_STATES = frozenset({*ACTIVE_SNAPSHOT_STATES, DEFAULTED})
DELINQUENT_STATES = frozenset({DPD_30, DPD_60, DPD_90_PLUS})

CURE_NUMERIC_FEATURES = (
    "loan_age",
    "current_actual_upb",
    "current_interest_rate",
    "estimated_ltv",
    "orig_classic_fico",
    "orig_original_ltv",
    "orig_original_dti",
)
CURE_CATEGORICAL_FEATURES = (
    "start_state",
    "vintage",
    "orig_occupancy_status",
    "orig_property_type",
    "orig_loan_purpose",
)
SEVERITY_NUMERIC_FEATURES = (
    "loan_age",
    "estimated_ltv",
    "orig_classic_fico",
    "orig_original_ltv",
    "orig_original_dti",
    "orig_original_interest_rate",
    "orig_original_upb",
)
SEVERITY_CATEGORICAL_FEATURES = (
    "vintage",
    "orig_occupancy_status",
    "orig_property_type",
    "orig_loan_purpose",
)

MODEL_PANEL_FEATURES = tuple(
    dict.fromkeys(
        [
            *[column for column in CURE_NUMERIC_FEATURES if column != "start_state"],
            *[column for column in CURE_CATEGORICAL_FEATURES if column != "start_state"],
            *SEVERITY_NUMERIC_FEATURES,
            *SEVERITY_CATEGORICAL_FEATURES,
        ]
    )
)

RECONCILIATION_COLUMNS = (
    "actual_loss",
    "zero_balance_removal_upb",
    "net_sales_proceeds",
    "delinquent_accrued_interest",
    "total_expenses",
    "mi_recoveries",
    "non_mi_recoveries",
)

PHASE3_PANEL_COLUMNS = tuple(
    dict.fromkeys(
        [
            "loan_id",
            "vintage",
            "reporting_month",
            "state",
            "delinquency_status",
            "zero_balance_code",
            "modification_flag",
            "current_actual_upb",
            "current_interest_bearing_upb",
            "current_non_interest_bearing_upb",
            "current_interest_rate",
            "remaining_months_to_legal_maturity",
            "delinquent_accrued_interest",
            *MODEL_PANEL_FEATURES,
            *RECONCILIATION_COLUMNS,
        ]
    )
)

_UNAVAILABLE_SENTINELS: Mapping[str, frozenset[float]] = {
    "orig_classic_fico": frozenset({9999.0}),
    "orig_original_ltv": frozenset({999.0}),
    "orig_original_dti": frozenset({999.0}),
    "estimated_ltv": frozenset({999.0}),
}


@dataclass(frozen=True)
class Phase3Settings:
    """Resolved Phase 3 settings with validated monthly cutoffs."""

    vintages: tuple[str, ...]
    development_end_month: int
    validation_end_month: int
    snapshot_month: int
    cure_consecutive_current_months: int
    cure_horizon_months: int
    maximum_iterations: int
    random_seed: int
    maximum_projection_months: int
    projection_sample_loans: int
    reuse_completed_quarters: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_month(value: Any, *, name: str) -> int:
    """Return a validated YYYYMM integer."""

    text = str(value).strip()
    if len(text) != 6 or not text.isdigit():
        raise ValueError(f"{name} must be a YYYYMM value")
    month = int(text[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"{name} must contain a valid month")
    return int(text)


def month_ordinal(value: Any) -> int:
    """Convert a YYYYMM value to a consecutive integer month index."""

    parsed = _validate_month(value, name="month")
    return (parsed // 100) * 12 + (parsed % 100) - 1


def add_months(value: Any, months: int) -> int:
    """Add whole calendar months to a YYYYMM value."""

    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be an integer")
    ordinal = month_ordinal(value) + months
    year, month_zero = divmod(ordinal, 12)
    return year * 100 + month_zero + 1


def resolve_phase3_settings(config: ProjectConfig) -> Phase3Settings:
    """Resolve optional Phase 3 settings without requiring a config edit."""

    phase1 = config.raw.get("phase1", {})
    modeling = config.raw.get("modeling", {})
    ifrs9 = config.raw.get("ifrs9", {})
    phase3 = config.raw.get("phase3", {})
    vintages = tuple(str(value).upper() for value in phase1.get("vintages", ()))
    if not vintages:
        raise ValueError("phase1.vintages must contain at least one quarter")
    development_end = _validate_month(
        modeling.get("development_end_month", 201112),
        name="development_end_month",
    )
    validation_end = _validate_month(
        modeling.get("validation_end_month", 201212),
        name="validation_end_month",
    )
    snapshot = _validate_month(
        modeling.get("backtest_snapshot_month", validation_end),
        name="backtest_snapshot_month",
    )
    if development_end >= validation_end:
        raise ValueError("development_end_month must precede validation_end_month")
    consecutive = int(phase3.get("cure_consecutive_current_months", 3))
    horizon = int(phase3.get("cure_horizon_months", 12))
    maximum_iterations = int(
        phase3.get("maximum_iterations", modeling.get("maximum_iterations", 1000))
    )
    maximum_projection = int(
        phase3.get(
            "maximum_projection_months",
            ifrs9.get("maximum_projection_months", 360),
        )
    )
    projection_sample = int(phase3.get("ead_projection_sample_loans", 250))
    if min(consecutive, horizon, maximum_iterations, maximum_projection) < 1:
        raise ValueError("Phase 3 positive integer settings must be at least one")
    if projection_sample < 0:
        raise ValueError("ead_projection_sample_loans must be nonnegative")
    return Phase3Settings(
        vintages=vintages,
        development_end_month=development_end,
        validation_end_month=validation_end,
        snapshot_month=snapshot,
        cure_consecutive_current_months=consecutive,
        cure_horizon_months=horizon,
        maximum_iterations=maximum_iterations,
        random_seed=int(config.raw.get("project", {}).get("random_seed", 42)),
        maximum_projection_months=maximum_projection,
        projection_sample_loans=projection_sample,
        reuse_completed_quarters=bool(
            phase3.get(
                "reuse_completed_quarters",
                phase1.get("reuse_completed_quarters", True),
            )
        ),
    )


def _normalise_zero_balance_code(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(2) if text.isdigit() else text.upper()


def validate_release47_terminal_mapping(panel: pd.DataFrame) -> dict[str, int]:
    """Validate the two Release 47 terminal codes critical to Phase 3."""

    required = ["zero_balance_code", "state"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing terminal mapping columns: {missing}")
    codes = panel["zero_balance_code"].map(_normalise_zero_balance_code)
    code15 = codes.eq("15")
    code16 = codes.eq(RPL_CENSOR_CODE)
    code15_bad = code15 & panel["state"].ne(DEFAULTED)
    code16_bad = code16 & panel["state"].ne(CENSORED_RPL)
    diagnostics = {
        "code15_default_rows": int(code15.sum()),
        "code15_misclassified_rows": int(code15_bad.sum()),
        "code16_censor_rows": int(code16.sum()),
        "code16_misclassified_rows": int(code16_bad.sum()),
    }
    if diagnostics["code15_misclassified_rows"]:
        raise ValueError("Release 47 zero balance code 15 must map to DEFAULTED")
    if diagnostics["code16_misclassified_rows"]:
        raise ValueError("Release 47 zero balance code 16 must map to CENSORED_RPL")
    return diagnostics


def numeric_series(values: pd.Series, *, column: str | None = None) -> pd.Series:
    """Parse a numeric source series and mask documented unavailable values."""

    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    if column in _UNAVAILABLE_SENTINELS:
        numeric = numeric.mask(numeric.isin(_UNAVAILABLE_SENTINELS[column]))
    return numeric


def clean_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stable numeric and categorical model types."""

    result = frame.copy()
    numeric_columns = set(CURE_NUMERIC_FEATURES) | set(SEVERITY_NUMERIC_FEATURES)
    numeric_columns.update(
        {
            "remaining_months_to_legal_maturity",
            "current_interest_bearing_upb",
            "current_non_interest_bearing_upb",
            "delinquent_accrued_interest",
        }
    )
    for column in numeric_columns:
        if column not in result:
            result[column] = np.nan
        result[column] = numeric_series(result[column], column=column)
    categorical_columns = set(CURE_CATEGORICAL_FEATURES) | set(
        SEVERITY_CATEGORICAL_FEATURES
    )
    for column in categorical_columns:
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        else:
            result[column] = result[column].astype("string").replace("", pd.NA)
    return result


def _period_to_yyyymm(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: np.nan
        if pd.isna(value)
        else int(value.year) * 100 + int(value.month)
    )


def prepare_cure_datasets(
    panel: pd.DataFrame,
    *,
    development_end_month: int,
    validation_end_month: int,
    expansion_weight: float,
    definition: CureDefinition,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build leakage-safe development and out-of-time cure episodes.

    Development labels are built from a panel physically truncated at the
    development cutoff. Validation labels are built from a panel truncated at
    the validation cutoff, then restricted to episodes beginning after the
    development cutoff. Censored targets remain in the saved dataset but are
    excluded by the model fitting function.
    """

    required = ["loan_id", "reporting_month", "state"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing cure panel columns: {missing}")
    work = panel.copy()
    for column in MODEL_PANEL_FEATURES:
        if column not in work:
            work[column] = pd.NA
    months = pd.to_numeric(work["reporting_month"], errors="coerce")
    if months.isna().any():
        raise ValueError("cure panel contains an invalid reporting month")
    development_panel = work.loc[months.le(development_end_month)].copy()
    validation_panel = work.loc[
        months.ge(development_end_month) & months.le(validation_end_month)
    ].copy()
    development_result = build_delinquency_episodes(
        development_panel,
        definition=definition,
        feature_cols=MODEL_PANEL_FEATURES,
    )
    validation_result = build_delinquency_episodes(
        validation_panel,
        definition=definition,
        feature_cols=MODEL_PANEL_FEATURES,
    )
    development = development_result.episodes.copy()
    validation = validation_result.episodes.copy()
    if len(development):
        development["episode_start_month"] = _period_to_yyyymm(
            development["start_month"]
        ).astype("Int64")
    else:
        development["episode_start_month"] = pd.Series(dtype="Int64")
    if len(validation):
        validation["episode_start_month"] = _period_to_yyyymm(
            validation["start_month"]
        ).astype("Int64")
        validation = validation.loc[
            validation["episode_start_month"].gt(development_end_month)
            & validation["episode_start_month"].le(validation_end_month)
            & ~validation["left_censored_start"]
        ].copy()
    else:
        validation["episode_start_month"] = pd.Series(dtype="Int64")
    development["split"] = "development"
    validation["split"] = "validation"
    episodes = pd.concat([development, validation], ignore_index=True, sort=False)
    episodes["sample_weight"] = float(expansion_weight)
    episodes = clean_model_features(episodes)
    diagnostics = {
        "definition": {
            "sustained_current_months": definition.consecutive_current_months,
            "outcome_horizon_months": definition.outcome_horizon_months,
            "censor_states": list(definition.censor_states),
            "default_states": list(definition.default_states),
        },
        "development": development_result.diagnostics.to_dict(),
        "validation_panel": validation_result.diagnostics.to_dict(),
        "saved_development_episodes": int(episodes["split"].eq("development").sum()),
        "saved_validation_episodes": int(episodes["split"].eq("validation").sum()),
        "development_observed_targets": int(
            episodes.loc[episodes["split"].eq("development"), "cured"].notna().sum()
        ),
        "validation_observed_targets": int(
            episodes.loc[episodes["split"].eq("validation"), "cured"].notna().sum()
        ),
    }
    return episodes, diagnostics


def extract_realized_lgd_rows(
    panel: pd.DataFrame,
    *,
    development_end_month: int,
    validation_end_month: int,
    expansion_weight: float,
) -> pd.DataFrame:
    """Extract reconciled disposition rows and preserve raw realized LGD."""

    required = [
        "loan_id",
        "reporting_month",
        "state",
        "zero_balance_code",
        *RECONCILIATION_COLUMNS,
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing realized LGD columns: {missing}")
    work = panel.copy()
    for column in MODEL_PANEL_FEATURES:
        if column not in work:
            work[column] = pd.NA
    codes = work["zero_balance_code"].map(_normalise_zero_balance_code)
    disclosed = work["actual_loss"].notna() & work["actual_loss"].astype(str).str.strip().ne("")
    selected = codes.isin(DEFAULT_ZERO_BALANCE_CODES) | disclosed
    losses = work.loc[selected].copy()
    if losses.empty:
        columns = [
            *required,
            "zero_balance_code_normalized",
            "resolved_default",
            "sample_weight",
            "split",
            "release47_status",
            "release47_difference",
            "release47_absolute_difference",
            "release47_recomputed_loss",
            "realized_lgd_raw",
            "realized_lgd_model",
        ]
        return pd.DataFrame(columns=columns)
    losses["zero_balance_code_normalized"] = codes.loc[selected].to_numpy()
    losses["resolved_default"] = losses["zero_balance_code_normalized"].isin(
        DEFAULT_ZERO_BALANCE_CODES
    )
    reconciliations = [
        reconcile_actual_loss(record)
        for record in losses.loc[:, RECONCILIATION_COLUMNS].to_dict("records")
    ]
    losses["release47_status"] = [item.status for item in reconciliations]
    losses["release47_difference"] = [
        None if item.difference is None else float(item.difference)
        for item in reconciliations
    ]
    losses["release47_absolute_difference"] = [
        None if item.absolute_difference is None else float(item.absolute_difference)
        for item in reconciliations
    ]
    losses["release47_recomputed_loss"] = [
        None if item.recomputed_loss is None else float(item.recomputed_loss)
        for item in reconciliations
    ]
    losses["actual_loss"] = numeric_series(losses["actual_loss"])
    losses["zero_balance_removal_upb"] = numeric_series(
        losses["zero_balance_removal_upb"]
    )
    losses = add_realized_lgd(losses, out_of_range="exclude")
    event_month = pd.to_numeric(losses["reporting_month"], errors="coerce")
    losses["event_month"] = event_month.astype("Int64")
    losses["split"] = np.select(
        [event_month.le(development_end_month), event_month.le(validation_end_month)],
        ["development", "validation"],
        default="outside_model_window",
    )
    losses["sample_weight"] = float(expansion_weight)
    losses = clean_model_features(losses)
    return losses.sort_values(["loan_id", "event_month"], kind="stable").reset_index(
        drop=True
    )


def _balance_basis(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    interest_bearing = numeric_series(
        frame.get("current_interest_bearing_upb", pd.Series(np.nan, index=frame.index))
    )
    actual_upb = numeric_series(
        frame.get("current_actual_upb", pd.Series(np.nan, index=frame.index))
    )
    non_interest = numeric_series(
        frame.get(
            "current_non_interest_bearing_upb", pd.Series(np.nan, index=frame.index)
        )
    )
    has_component = interest_bearing.notna()
    principal = interest_bearing.where(has_component, actual_upb)
    non_interest_used = non_interest.fillna(0.0).where(has_component, 0.0)
    basis = pd.Series(
        np.where(has_component, "component_fields", "current_actual_upb_fallback"),
        index=frame.index,
        dtype="string",
    )
    return principal, non_interest_used, basis


def _fico_segment(values: pd.Series) -> pd.Series:
    numeric = numeric_series(values, column="orig_classic_fico")
    return pd.cut(
        numeric,
        bins=[-np.inf, 659, 699, 739, np.inf],
        labels=["Below 660", "660 to 699", "700 to 739", "740 plus"],
    ).astype("string").fillna("Unknown")


def _ltv_segment(values: pd.Series) -> pd.Series:
    numeric = numeric_series(values, column="orig_original_ltv")
    return pd.cut(
        numeric,
        bins=[-np.inf, 60, 80, 90, np.inf],
        labels=["60 or below", "61 to 80", "81 to 90", "Above 90"],
    ).astype("string").fillna("Unknown")


def prepare_snapshot_inputs(
    panel: pd.DataFrame,
    *,
    snapshot_month: int,
    expansion_weight: float,
) -> pd.DataFrame:
    """Create one active loan row with auditable scalar EAD at the snapshot."""

    required = ["loan_id", "reporting_month", "state"]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing snapshot columns: {missing}")
    months = pd.to_numeric(panel["reporting_month"], errors="coerce")
    snapshot = panel.loc[
        months.eq(snapshot_month) & panel["state"].isin(REPORTING_SNAPSHOT_STATES)
    ].copy()
    if snapshot.empty:
        return snapshot.assign(
            sample_weight=pd.Series(dtype="float64"),
            ead=pd.Series(dtype="float64"),
        )
    duplicated = snapshot["loan_id"].duplicated(keep=False)
    if duplicated.any():
        examples = snapshot.loc[duplicated, "loan_id"].head(5).tolist()
        raise ValueError(f"snapshot contains duplicate loan rows: {examples}")
    snapshot = clean_model_features(snapshot)
    principal, non_interest, basis = _balance_basis(snapshot)
    removal_upb = numeric_series(
        snapshot.get("zero_balance_removal_upb", pd.Series(np.nan, index=snapshot.index))
    )
    disclosed_default_balance = snapshot["state"].eq(DEFAULTED) & removal_upb.gt(0)
    principal = principal.where(~disclosed_default_balance, removal_upb)
    non_interest = non_interest.where(~disclosed_default_balance, 0.0)
    basis = basis.where(
        ~disclosed_default_balance, "zero_balance_removal_upb_default"
    )
    accrued = numeric_series(snapshot["delinquent_accrued_interest"]).fillna(0.0)
    valid = (
        principal.notna()
        & principal.ge(0)
        & non_interest.ge(0)
        & accrued.ge(0)
    )
    ead = pd.Series(np.nan, index=snapshot.index, dtype="float64")
    for index in snapshot.index[valid]:
        ead.loc[index] = calculate_default_ead(
            principal.loc[index],
            non_interest_bearing_upb=non_interest.loc[index],
            delinquent_accrued_interest=accrued.loc[index],
        )
    rates_percent = numeric_series(snapshot["current_interest_rate"])
    snapshot["ead_interest_bearing_upb"] = principal
    snapshot["ead_non_interest_bearing_upb"] = non_interest
    snapshot["ead_delinquent_accrued_interest"] = accrued
    snapshot["ead_balance_basis"] = basis
    snapshot["ead_component_valid"] = valid
    snapshot["ead"] = ead
    snapshot["effective_annual_rate"] = rates_percent / 100.0
    snapshot["sample_weight"] = float(expansion_weight)
    snapshot["start_state"] = snapshot["state"].astype("string")
    snapshot["delinquency_segment"] = snapshot["state"].astype("string")
    snapshot["fico_segment"] = _fico_segment(snapshot["orig_classic_fico"])
    snapshot["ltv_segment"] = _ltv_segment(snapshot["orig_original_ltv"])
    snapshot["segment"] = (
        snapshot["vintage"].astype("string")
        + "|"
        + snapshot["fico_segment"]
        + "|"
        + snapshot["ltv_segment"]
    )
    return snapshot.reset_index(drop=True)


def build_ead_validation_rows(
    panel: pd.DataFrame,
    *,
    validation_start_month: int,
    validation_end_month: int,
    expansion_weight: float,
) -> pd.DataFrame:
    """Compare one month scheduled principal with clean out-of-time observations."""

    required = [
        "loan_id",
        "reporting_month",
        "state",
        "current_actual_upb",
        "current_interest_bearing_upb",
        "current_non_interest_bearing_upb",
        "current_interest_rate",
        "remaining_months_to_legal_maturity",
        "modification_flag",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing EAD validation columns: {missing}")
    work = panel.loc[:, required].copy()
    months = pd.to_numeric(work["reporting_month"], errors="coerce")
    if months.isna().any():
        raise ValueError("EAD panel contains an invalid reporting month")
    work["reporting_month"] = months.astype("int64")
    work["_ordinal"] = work["reporting_month"].map(month_ordinal)
    principal, non_interest, basis = _balance_basis(work)
    work["_principal"] = principal
    work["_non_interest"] = non_interest
    work["_balance_basis"] = basis
    work["_rate"] = numeric_series(work["current_interest_rate"])
    work["_term"] = numeric_series(work["remaining_months_to_legal_maturity"])
    work = work.sort_values(["loan_id", "_ordinal"], kind="stable")
    grouped = work.groupby("loan_id", sort=False, observed=True)
    for column in (
        "_ordinal",
        "_principal",
        "_non_interest",
        "state",
        "modification_flag",
    ):
        work[f"_next_{column}"] = grouped[column].shift(-1)
    adjacent = work["_next__ordinal"].sub(work["_ordinal"]).eq(1)
    in_window = work["reporting_month"].between(
        validation_start_month, validation_end_month
    )
    current_active = work["state"].isin(ACTIVE_SNAPSHOT_STATES)
    next_active = work["_next_state"].isin(ACTIVE_SNAPSHOT_STATES)
    unmodified = (
        work["modification_flag"].fillna("").astype(str).str.upper().ne("Y")
        & work["_next_modification_flag"].fillna("").astype(str).str.upper().ne("Y")
    )
    numeric_valid = (
        work["_principal"].ge(0)
        & work["_next__principal"].ge(0)
        & work["_non_interest"].ge(0)
        & work["_next__non_interest"].ge(0)
        & work["_rate"].ge(0)
        & work["_term"].ge(1)
    )
    selected = work.loc[
        adjacent & in_window & current_active & next_active & unmodified & numeric_valid
    ].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "reporting_month",
                "scheduled_next_balance",
                "observed_next_balance",
                "error",
                "absolute_error",
                "squared_error",
                "sample_weight",
            ]
        )
    predicted = [
        scheduled_balance_at_month(principal_value, rate, int(term), 1, rate_in_percent=True)
        for principal_value, rate, term in zip(
            selected["_principal"], selected["_rate"], selected["_term"], strict=True
        )
    ]
    predicted_total = np.asarray(predicted) + selected["_non_interest"].to_numpy()
    observed_total = (
        selected["_next__principal"] + selected["_next__non_interest"]
    ).to_numpy(dtype="float64")
    error = predicted_total - observed_total
    result = pd.DataFrame(
        {
            "loan_id": selected["loan_id"].to_numpy(),
            "vintage": selected.get("vintage", pd.Series("", index=selected.index)).to_numpy(),
            "reporting_month": selected["reporting_month"].to_numpy(),
            "current_balance": (
                selected["_principal"] + selected["_non_interest"]
            ).to_numpy(),
            "scheduled_next_balance": predicted_total,
            "observed_next_balance": observed_total,
            "error": error,
            "absolute_error": np.abs(error),
            "squared_error": np.square(error),
            "balance_basis": selected["_balance_basis"].to_numpy(),
            "sample_weight": float(expansion_weight),
        }
    )
    return result


def evaluate_cure_predictions(
    model: WeightedCureModel, data: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted cure model on observed targets with sampling weights."""

    observed = data.loc[data["cured"].notna()].copy()
    if observed.empty:
        return (
            {"rows": 0, "weight_sum": 0.0, "roc_auc": None, "log_loss": None, "brier_score": None},
            observed,
            pd.DataFrame(),
        )
    target = pd.to_numeric(observed["cured"], errors="raise").to_numpy(dtype="int8")
    weights = numeric_series(observed["sample_weight"]).to_numpy(dtype="float64")
    probabilities = model.predict_probability(observed)
    observed["predicted_cure_probability"] = probabilities
    auc = (
        float(roc_auc_score(target, probabilities, sample_weight=weights))
        if np.unique(target).size == 2
        else None
    )
    metrics = {
        "rows": len(observed),
        "positive_rows": int(target.sum()),
        "negative_rows": int((target == 0).sum()),
        "weight_sum": float(weights.sum()),
        "weighted_observed_rate": float(np.average(target, weights=weights)),
        "weighted_predicted_rate": float(np.average(probabilities, weights=weights)),
        "roc_auc": auc,
        "log_loss": float(log_loss(target, probabilities, sample_weight=weights, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities, sample_weight=weights)),
    }
    bins = pd.cut(
        probabilities,
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        duplicates="drop",
    )
    calibration_rows = []
    for bin_label, indexes in observed.groupby(bins, observed=True).groups.items():
        positions = observed.index.get_indexer(indexes)
        bin_weights = weights[positions]
        calibration_rows.append(
            {
                "probability_bin": str(bin_label),
                "rows": len(positions),
                "weight_sum": float(bin_weights.sum()),
                "weighted_predicted": float(
                    np.average(probabilities[positions], weights=bin_weights)
                ),
                "weighted_observed": float(np.average(target[positions], weights=bin_weights)),
            }
        )
    return metrics, observed, pd.DataFrame(calibration_rows)


def evaluate_severity_predictions(
    model: ConditionalSeverityModel, data: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate conditional severity on eligible out-of-time defaults."""

    eligible = data.loc[
        data["resolved_default"].fillna(False)
        & data["realized_lgd_model"].notna()
    ].copy()
    if eligible.empty:
        return (
            {"rows": 0, "weight_sum": 0.0, "mae": None, "rmse": None},
            eligible,
            pd.DataFrame(),
        )
    observed = numeric_series(eligible["realized_lgd_model"]).to_numpy(dtype="float64")
    weights = numeric_series(eligible["sample_weight"]).to_numpy(dtype="float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predicted = model.predict(eligible)
    eligible["predicted_conditional_lgd"] = predicted
    error = predicted - observed
    metrics = {
        "rows": len(eligible),
        "weight_sum": float(weights.sum()),
        "weighted_observed_mean": float(np.average(observed, weights=weights)),
        "weighted_predicted_mean": float(np.average(predicted, weights=weights)),
        "mean_error": float(np.average(error, weights=weights)),
        "mae": float(np.average(np.abs(error), weights=weights)),
        "rmse": float(np.sqrt(np.average(np.square(error), weights=weights))),
    }
    bins = pd.cut(
        predicted,
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        duplicates="drop",
    )
    calibration_rows = []
    for bin_label, indexes in eligible.groupby(bins, observed=True).groups.items():
        positions = eligible.index.get_indexer(indexes)
        bin_weights = weights[positions]
        calibration_rows.append(
            {
                "prediction_bin": str(bin_label),
                "rows": len(positions),
                "weight_sum": float(bin_weights.sum()),
                "weighted_predicted": float(np.average(predicted[positions], weights=bin_weights)),
                "weighted_observed": float(np.average(observed[positions], weights=bin_weights)),
            }
        )
    return metrics, eligible, pd.DataFrame(calibration_rows)


def evaluate_ead_validation(data: pd.DataFrame) -> dict[str, Any]:
    """Return weighted scheduled balance validation metrics and sufficient sums."""

    if data.empty:
        return {
            "rows": 0,
            "weight_sum": 0.0,
            "weighted_error_sum": 0.0,
            "weighted_absolute_error_sum": 0.0,
            "weighted_squared_error_sum": 0.0,
            "mean_error": None,
            "mae": None,
            "rmse": None,
        }
    weights = numeric_series(data["sample_weight"]).to_numpy(dtype="float64")
    error = numeric_series(data["error"]).to_numpy(dtype="float64")
    weighted_error = float(np.dot(weights, error))
    weighted_absolute = float(np.dot(weights, np.abs(error)))
    weighted_squared = float(np.dot(weights, np.square(error)))
    weight_sum = float(weights.sum())
    return {
        "rows": len(data),
        "weight_sum": weight_sum,
        "weighted_error_sum": weighted_error,
        "weighted_absolute_error_sum": weighted_absolute,
        "weighted_squared_error_sum": weighted_squared,
        "mean_error": weighted_error / weight_sum,
        "mae": weighted_absolute / weight_sum,
        "rmse": math.sqrt(weighted_squared / weight_sum),
    }


def _stable_hash(value: str, *, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def stable_row_sample(
    data: pd.DataFrame,
    maximum_rows: int,
    *,
    id_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Select the smallest stable hashes without depending on input order."""

    if maximum_rows < 0:
        raise ValueError("maximum_rows must be nonnegative")
    if maximum_rows == 0 or data.empty:
        return data.iloc[0:0].copy()
    missing = [column for column in id_columns if column not in data]
    if missing:
        raise KeyError(f"missing stable sample columns: {missing}")
    if len(data) <= maximum_rows:
        return data.copy().reset_index(drop=True)
    keys = data.loc[:, id_columns].astype(str).agg("|".join, axis=1)
    scores = keys.map(lambda value: _stable_hash(value, seed=seed))
    selected = scores.nsmallest(maximum_rows).index
    return data.loc[selected].copy().reset_index(drop=True)


def project_snapshot_ead_sample(
    snapshot: pd.DataFrame,
    *,
    maximum_loans: int,
    maximum_projection_months: int,
    seed: int,
) -> pd.DataFrame:
    """Create auditable amortizing EAD curves for a deterministic loan sample."""

    eligible = snapshot.loc[
        snapshot["ead_component_valid"].fillna(False)
        & snapshot["state"].isin(ACTIVE_SNAPSHOT_STATES)
        & snapshot["current_interest_rate"].notna()
        & snapshot["remaining_months_to_legal_maturity"].notna()
    ].copy()
    eligible = eligible.loc[eligible["remaining_months_to_legal_maturity"].ge(1)]
    selected = stable_row_sample(
        eligible,
        maximum_loans,
        id_columns=["loan_id"],
        seed=seed,
    )
    frames = []
    for row in selected.itertuples(index=False):
        term = int(row.remaining_months_to_legal_maturity)
        horizon = min(term, maximum_projection_months)
        projection = project_amortizing_ead(
            row.ead_interest_bearing_upb,
            row.current_interest_rate,
            term,
            horizon_months=horizon,
            rate_in_percent=True,
            non_interest_bearing_upb=row.ead_non_interest_bearing_upb,
            delinquent_accrued_interest=row.ead_delinquent_accrued_interest,
        )
        projection.insert(0, "loan_id", row.loan_id)
        projection.insert(1, "vintage", row.vintage)
        projection["sample_weight"] = row.sample_weight
        projection["effective_annual_rate"] = row.effective_annual_rate
        frames.append(projection)
    if not frames:
        return pd.DataFrame(
            columns=[
                "loan_id",
                "vintage",
                "month",
                "interest_bearing_upb",
                "non_interest_bearing_upb",
                "delinquent_accrued_interest",
                "default_ead",
                "sample_weight",
                "effective_annual_rate",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _quarter_source_paths(config: ProjectConfig, vintage: str) -> tuple[Path, Path]:
    key = vintage.lower()
    panel = config.path("processed_data") / "phase1" / f"panel_{key}.parquet"
    summary = config.path("artifacts") / "phase1" / "quarters" / f"{key}.json"
    if not panel.is_file():
        raise FileNotFoundError(f"Phase 1 panel is missing: {panel}")
    if not summary.is_file():
        raise FileNotFoundError(f"Phase 1 quarter summary is missing: {summary}")
    return panel, summary


def _quarter_checkpoint_paths(config: ProjectConfig, vintage: str) -> dict[str, Path]:
    key = vintage.lower()
    directory = config.path("processed_data") / "phase3" / "quarters"
    metadata = config.path("artifacts") / "phase3" / "quarters"
    return {
        "cure": directory / f"cure_{key}.parquet",
        "loss": directory / f"loss_{key}.parquet",
        "snapshot": directory / f"snapshot_{key}.parquet",
        "ead_validation": directory / f"ead_validation_{key}.parquet",
        "metadata": metadata / f"{key}.json",
    }


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
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    joblib.dump(value, temporary)
    temporary.replace(path)


def _source_manifest(config: ProjectConfig, settings: Phase3Settings) -> list[dict[str, Any]]:
    manifest = []
    for vintage in settings.vintages:
        panel, summary_path = _quarter_source_paths(config, vintage)
        with summary_path.open(encoding="utf-8") as handle:
            quarter = json.load(handle)
        manifest.append(
            {
                "vintage": vintage,
                "panel": str(panel),
                "panel_bytes": panel.stat().st_size,
                "panel_rows": pq.ParquetFile(panel).metadata.num_rows,
                "release": quarter.get("release"),
                "sample_seed": quarter.get("sample_seed"),
                "sampled_loans": quarter.get("sampled_loans"),
                "population_originations": quarter.get("population_originations"),
                "expansion_weight": quarter.get("expansion_weight"),
            }
        )
    return manifest


def phase3_run_signature(
    settings: Phase3Settings, source_manifest: Sequence[Mapping[str, Any]]
) -> str:
    """Return a deterministic signature for checkpoint validation."""

    payload = {
        "pipeline_version": 2,
        "settings": settings.to_dict(),
        "sources": list(source_manifest),
        "cure_numeric_features": CURE_NUMERIC_FEATURES,
        "cure_categorical_features": CURE_CATEGORICAL_FEATURES,
        "severity_numeric_features": SEVERITY_NUMERIC_FEATURES,
        "severity_categorical_features": SEVERITY_CATEGORICAL_FEATURES,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint_metadata(
    paths: Mapping[str, Path], *, run_signature: str
) -> dict[str, Any] | None:
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        with paths["metadata"].open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("run_signature") != run_signature:
        return None
    expected_rows = metadata.get("rows", {})
    for name in ("cure", "loss", "snapshot", "ead_validation"):
        if pq.ParquetFile(paths[name]).metadata.num_rows != expected_rows.get(name):
            return None
    return metadata


def _process_quarter(
    config: ProjectConfig,
    settings: Phase3Settings,
    source: Mapping[str, Any],
    *,
    run_signature: str,
) -> dict[str, Any]:
    vintage = str(source["vintage"])
    paths = _quarter_checkpoint_paths(config, vintage)
    if settings.reuse_completed_quarters:
        reusable = _load_checkpoint_metadata(paths, run_signature=run_signature)
        if reusable is not None:
            return reusable
    panel_path = Path(str(source["panel"]))
    read_cutoff = max(settings.validation_end_month, settings.snapshot_month)
    panel = pd.read_parquet(
        panel_path,
        columns=list(PHASE3_PANEL_COLUMNS),
        filters=[("reporting_month", "<=", str(read_cutoff))],
    )
    terminal_diagnostics = validate_release47_terminal_mapping(panel)
    definition = CureDefinition(
        consecutive_current_months=settings.cure_consecutive_current_months,
        outcome_horizon_months=settings.cure_horizon_months,
    )
    cure, cure_diagnostics = prepare_cure_datasets(
        panel,
        development_end_month=settings.development_end_month,
        validation_end_month=settings.validation_end_month,
        expansion_weight=float(source["expansion_weight"]),
        definition=definition,
    )
    losses = extract_realized_lgd_rows(
        panel,
        development_end_month=settings.development_end_month,
        validation_end_month=settings.validation_end_month,
        expansion_weight=float(source["expansion_weight"]),
    )
    snapshot = prepare_snapshot_inputs(
        panel,
        snapshot_month=settings.snapshot_month,
        expansion_weight=float(source["expansion_weight"]),
    )
    ead_validation = build_ead_validation_rows(
        panel,
        validation_start_month=add_months(settings.development_end_month, 1),
        validation_end_month=settings.validation_end_month,
        expansion_weight=float(source["expansion_weight"]),
    )
    del panel
    frames = {
        "cure": cure,
        "loss": losses,
        "snapshot": snapshot,
        "ead_validation": ead_validation,
    }
    for name, frame in frames.items():
        _atomic_parquet(frame, paths[name])
    reconciliation_counts = losses["release47_status"].value_counts().to_dict()
    metadata = {
        "run_signature": run_signature,
        "vintage": vintage,
        "expansion_weight": float(source["expansion_weight"]),
        "rows": {name: len(frame) for name, frame in frames.items()},
        "terminal_mapping": terminal_diagnostics,
        "cure": cure_diagnostics,
        "release47_reconciliation": {
            "status_counts": {str(key): int(value) for key, value in reconciliation_counts.items()},
            "maximum_absolute_difference": (
                None
                if losses.empty or losses["release47_absolute_difference"].dropna().empty
                else float(losses["release47_absolute_difference"].max())
            ),
        },
        "ead_validation": evaluate_ead_validation(ead_validation),
        "outputs": {name: str(path) for name, path in paths.items() if name != "metadata"},
    }
    _atomic_json(metadata, paths["metadata"])
    return metadata


def _concat_checkpoints(
    config: ProjectConfig, vintages: Iterable[str], name: str
) -> pd.DataFrame:
    frames = [
        pd.read_parquet(_quarter_checkpoint_paths(config, vintage)[name])
        for vintage in vintages
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _reconciliation_table(
    config: ProjectConfig,
    settings: Phase3Settings,
    quarter_metadata: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for metadata in quarter_metadata:
        counts = metadata["release47_reconciliation"]["status_counts"]
        terminal = metadata["terminal_mapping"]
        rows.append(
            {
                "vintage": metadata["vintage"],
                "reconciled": int(counts.get("reconciled", 0)),
                "outside_tolerance": int(counts.get("outside_tolerance", 0)),
                "actual_loss_missing": int(counts.get("actual_loss_missing", 0)),
                "maximum_absolute_difference": metadata["release47_reconciliation"][
                    "maximum_absolute_difference"
                ],
                **terminal,
            }
        )
    return pd.DataFrame(rows)


def _ead_validation_table(
    config: ProjectConfig,
    settings: Phase3Settings,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows = []
    samples = []
    totals = {
        "rows": 0,
        "weight_sum": 0.0,
        "weighted_error_sum": 0.0,
        "weighted_absolute_error_sum": 0.0,
        "weighted_squared_error_sum": 0.0,
    }
    for vintage in settings.vintages:
        frame = pd.read_parquet(_quarter_checkpoint_paths(config, vintage)["ead_validation"])
        metrics = evaluate_ead_validation(frame)
        rows.append({"vintage": vintage, **metrics})
        for key in totals:
            totals[key] += metrics[key]
        samples.append(
            stable_row_sample(
                frame,
                1000,
                id_columns=["loan_id", "reporting_month"],
                seed=settings.random_seed,
            )
        )
        del frame
    weight_sum = totals["weight_sum"]
    pooled = {
        **totals,
        "mean_error": (
            totals["weighted_error_sum"] / weight_sum if weight_sum else None
        ),
        "mae": (
            totals["weighted_absolute_error_sum"] / weight_sum if weight_sum else None
        ),
        "rmse": (
            math.sqrt(totals["weighted_squared_error_sum"] / weight_sum)
            if weight_sum
            else None
        ),
    }
    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    return pd.DataFrame(rows), pooled, sample


def apply_cure_adjusted_lgd(data: pd.DataFrame) -> pd.DataFrame:
    """Combine cure and conditional severity into the reporting LGD input.

    Delinquent but nonterminal reporting rows use one minus cure probability
    times conditional severity. CURRENT rows have no active delinquency episode,
    so conditional severity is retained. Terminal DEFAULTED rows are already in
    default, so cure is not applicable and conditional severity is retained.
    The 90_PLUS state is nonterminal in the Freddie panel and receives the same
    explicit cure adjustment as 30 and 60 DPD.
    """

    required = ["state", "conditional_lgd", "cure_probability"]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"missing cure adjusted LGD columns: {missing}")
    result = data.copy()
    severity = numeric_series(result["conditional_lgd"])
    cure = numeric_series(result["cure_probability"])
    applicable = result["state"].isin(DELINQUENT_STATES)
    missing_cure = applicable & cure.isna()
    if missing_cure.any():
        examples = result.loc[missing_cure, "state"].head(5).tolist()
        raise ValueError(f"delinquent rows require cure probability: {examples}")
    if ((cure.dropna() < 0) | (cure.dropna() > 1)).any():
        raise ValueError("cure probability must be between zero and one")
    result["cure_adjustment_applicable"] = applicable
    result["expected_lgd"] = severity.where(~applicable, (1.0 - cure) * severity)
    result["lgd_formula"] = np.select(
        [
            applicable,
            result["state"].eq(DEFAULTED),
            result["state"].eq(CURRENT),
        ],
        [
            "(1 - cure_probability) * conditional_lgd",
            "conditional_lgd; terminal default, cure not applicable",
            "conditional_lgd; no active delinquency episode",
        ],
        default="conditional_lgd; cure not applicable",
    )
    return result


def _adapter_frames(
    snapshot: pd.DataFrame,
    cure_model: WeightedCureModel,
    severity_model: ConditionalSeverityModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched = snapshot.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        enriched["conditional_lgd"] = severity_model.predict(enriched)
    enriched["cure_probability"] = np.nan
    delinquent = enriched["state"].isin(DELINQUENT_STATES)
    if delinquent.any():
        enriched.loc[delinquent, "cure_probability"] = cure_model.predict_probability(
            enriched.loc[delinquent]
        )
    enriched.loc[enriched["state"].eq(DEFAULTED), "cure_probability"] = 0.0
    enriched = apply_cure_adjusted_lgd(enriched)
    enriched["lgd"] = enriched["expected_lgd"].clip(0.0, 1.0)
    enriched["predicted_lgd"] = enriched["lgd"]
    enriched["predicted_ead"] = enriched["ead"]
    common = [
        "loan_id",
        "vintage",
        "reporting_month",
        "state",
        "sample_weight",
        "segment",
        "fico_segment",
        "ltv_segment",
        "delinquency_segment",
        "effective_annual_rate",
    ]
    lgd_inputs = enriched.loc[
        :,
        [
            *common,
            "lgd",
            "predicted_lgd",
            "expected_lgd",
            "conditional_lgd",
            "cure_probability",
            "cure_adjustment_applicable",
            "lgd_formula",
        ],
    ].copy()
    ead_inputs = enriched.loc[
        :,
        [
            *common,
            "ead",
            "ead_interest_bearing_upb",
            "ead_non_interest_bearing_upb",
            "ead_delinquent_accrued_interest",
            "ead_balance_basis",
            "ead_component_valid",
            "current_interest_rate",
            "remaining_months_to_legal_maturity",
        ],
    ].copy()
    ead_inputs["predicted_ead"] = ead_inputs["ead"]
    snapshot_inputs = enriched.loc[
        :,
        [
            *common,
            "lgd",
            "predicted_lgd",
            "expected_lgd",
            "conditional_lgd",
            "cure_probability",
            "cure_adjustment_applicable",
            "lgd_formula",
            "ead",
            "predicted_ead",
            "ead_interest_bearing_upb",
            "ead_non_interest_bearing_upb",
            "ead_delinquent_accrued_interest",
            "remaining_months_to_legal_maturity",
            "orig_classic_fico",
            "orig_original_ltv",
            "orig_original_dti",
            "orig_occupancy_status",
            "orig_property_type",
            "orig_loan_purpose",
        ],
    ].copy()
    return snapshot_inputs, lgd_inputs, ead_inputs


def _save_phase3_figures(
    config: ProjectConfig,
    cure_calibration: pd.DataFrame,
    severity_calibration: pd.DataFrame,
    ead_sample: pd.DataFrame,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = config.path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if not cure_calibration.empty:
        figure, axis = plt.subplots(figsize=(5.5, 5.0))
        axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        axis.plot(
            cure_calibration["weighted_predicted"],
            cure_calibration["weighted_observed"],
            marker="o",
            color="#2f6b4f",
        )
        axis.set_xlabel("Weighted predicted cure probability")
        axis.set_ylabel("Weighted observed cure rate")
        axis.set_title("Out-of-time cure calibration")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = directory / "phase3_cure_calibration.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs["cure_calibration_figure"] = str(path)
    if not severity_calibration.empty:
        figure, axis = plt.subplots(figsize=(5.5, 5.0))
        axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        axis.plot(
            severity_calibration["weighted_predicted"],
            severity_calibration["weighted_observed"],
            marker="o",
            color="#8a4f2b",
        )
        axis.set_xlabel("Weighted predicted conditional LGD")
        axis.set_ylabel("Weighted observed realized LGD")
        axis.set_title("Out-of-time severity calibration")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = directory / "phase3_severity_calibration.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs["severity_calibration_figure"] = str(path)
    if not ead_sample.empty:
        relative = ead_sample.loc[ead_sample["observed_next_balance"].gt(0)].copy()
        relative["relative_error"] = (
            relative["error"] / relative["observed_next_balance"]
        ).clip(-0.25, 0.25)
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        axis.hist(relative["relative_error"], bins=60, color="#355f8a", alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_xlabel("Scheduled minus observed balance, divided by observed")
        axis.set_ylabel("Validation loan months")
        axis.set_title("Out-of-time one-month EAD projection errors")
        axis.grid(alpha=0.2, axis="y")
        figure.tight_layout()
        path = directory / "phase3_ead_validation_errors.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs["ead_validation_figure"] = str(path)
    return outputs


def _existing_completed_summary(
    summary_path: Path, *, run_signature: str
) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("run_signature") != run_signature:
        return None
    outputs = summary.get("outputs", {})
    required = (
        "cure_model",
        "severity_model",
        "snapshot_inputs",
        "lgd_inputs",
        "ead_inputs",
    )
    if not all(Path(str(outputs.get(key, ""))).is_file() for key in required):
        return None
    return summary


def run_phase3(config: ProjectConfig) -> dict[str, Any]:
    """Execute restart-safe cure, severity, and amortizing EAD integration."""

    settings = resolve_phase3_settings(config)
    config.ensure_output_directories()
    source_manifest = _source_manifest(config, settings)
    run_signature = phase3_run_signature(settings, source_manifest)
    summary_path = config.path("artifacts") / "phase3_summary.json"
    if settings.reuse_completed_quarters:
        existing = _existing_completed_summary(summary_path, run_signature=run_signature)
        if existing is not None:
            return existing

    quarter_metadata = []
    for source in source_manifest:
        vintage = str(source["vintage"])
        print(f"Phase 3 compact outcomes: {vintage}", flush=True)
        metadata = _process_quarter(
            config,
            settings,
            source,
            run_signature=run_signature,
        )
        quarter_metadata.append(metadata)
        print(
            f"Completed {vintage}: {metadata['rows']['cure']} cure episodes, "
            f"{metadata['rows']['loss']} disposition rows",
            flush=True,
        )

    processed = config.path("processed_data") / "phase3"
    artifact_directory = config.path("artifacts") / "phase3"
    model_directory = artifact_directory / "models"
    table_directory = config.path("artifacts") / "tables"
    cure_dataset = _concat_checkpoints(config, settings.vintages, "cure")
    loss_dataset = _concat_checkpoints(config, settings.vintages, "loss")
    snapshot = _concat_checkpoints(config, settings.vintages, "snapshot")
    _atomic_parquet(cure_dataset, processed / "cure_dataset.parquet")
    _atomic_parquet(loss_dataset, processed / "realized_lgd_dataset.parquet")

    development_cure = cure_dataset.loc[cure_dataset["split"].eq("development")]
    cure_model = fit_weighted_cure_model(
        development_cure,
        numeric_features=CURE_NUMERIC_FEATURES,
        categorical_features=CURE_CATEGORICAL_FEATURES,
        weight_col="sample_weight",
        max_iter=settings.maximum_iterations,
        random_state=settings.random_seed,
        raise_on_nonconvergence=True,
    )
    development_loss = loss_dataset.loc[loss_dataset["split"].eq("development")]
    severity_model = fit_conditional_severity_model(
        development_loss,
        numeric_features=SEVERITY_NUMERIC_FEATURES,
        categorical_features=SEVERITY_CATEGORICAL_FEATURES,
        condition_col="resolved_default",
        condition_value=True,
        weight_col="sample_weight",
        out_of_range="exclude",
        max_iter=settings.maximum_iterations,
        raise_on_nonconvergence=True,
    )
    cure_model_path = model_directory / "cure_model.joblib"
    severity_model_path = model_directory / "severity_model.joblib"
    _atomic_joblib(cure_model, cure_model_path)
    _atomic_joblib(severity_model, severity_model_path)

    validation_cure = cure_dataset.loc[cure_dataset["split"].eq("validation")]
    cure_metrics, cure_predictions, cure_calibration = evaluate_cure_predictions(
        cure_model, validation_cure
    )
    validation_loss = loss_dataset.loc[loss_dataset["split"].eq("validation")]
    severity_metrics, severity_predictions, severity_calibration = (
        evaluate_severity_predictions(severity_model, validation_loss)
    )
    _atomic_parquet(cure_predictions, processed / "cure_validation_predictions.parquet")
    _atomic_parquet(
        severity_predictions,
        processed / "severity_validation_predictions.parquet",
    )
    _atomic_csv(cure_calibration, table_directory / "phase3_cure_calibration.csv")
    _atomic_csv(
        severity_calibration,
        table_directory / "phase3_severity_calibration.csv",
    )

    snapshot_inputs, lgd_inputs, ead_inputs = _adapter_frames(
        snapshot, cure_model, severity_model
    )
    snapshot_path = artifact_directory / "snapshot_inputs.parquet"
    lgd_path = artifact_directory / "lgd_inputs.parquet"
    ead_path = artifact_directory / "ead_inputs.parquet"
    _atomic_parquet(snapshot_inputs, snapshot_path)
    _atomic_parquet(lgd_inputs, lgd_path)
    _atomic_parquet(ead_inputs, ead_path)
    processed_snapshot_path = processed / "snapshot_inputs.parquet"
    processed_lgd_path = processed / "lgd_inputs.parquet"
    processed_ead_path = processed / "ead_inputs.parquet"
    _atomic_parquet(snapshot_inputs, processed_snapshot_path)
    _atomic_parquet(lgd_inputs, processed_lgd_path)
    _atomic_parquet(ead_inputs, processed_ead_path)
    ead_projection = project_snapshot_ead_sample(
        snapshot,
        maximum_loans=settings.projection_sample_loans,
        maximum_projection_months=settings.maximum_projection_months,
        seed=settings.random_seed,
    )
    ead_projection_path = artifact_directory / "ead_projection_sample.parquet"
    _atomic_parquet(ead_projection, ead_projection_path)

    reconciliation = _reconciliation_table(config, settings, quarter_metadata)
    reconciliation_path = table_directory / "phase3_release47_reconciliation.csv"
    _atomic_csv(reconciliation, reconciliation_path)
    ead_by_vintage, ead_pooled, ead_sample = _ead_validation_table(config, settings)
    ead_validation_path = table_directory / "phase3_ead_validation.csv"
    _atomic_csv(ead_by_vintage, ead_validation_path)
    figures = _save_phase3_figures(
        config, cure_calibration, severity_calibration, ead_sample
    )

    outputs = {
        "cure_model": str(cure_model_path),
        "severity_model": str(severity_model_path),
        "cure_dataset": str(processed / "cure_dataset.parquet"),
        "realized_lgd_dataset": str(processed / "realized_lgd_dataset.parquet"),
        "cure_validation_predictions": str(
            processed / "cure_validation_predictions.parquet"
        ),
        "severity_validation_predictions": str(
            processed / "severity_validation_predictions.parquet"
        ),
        "snapshot_inputs": str(snapshot_path),
        "lgd_inputs": str(lgd_path),
        "ead_inputs": str(ead_path),
        "processed_snapshot_inputs": str(processed_snapshot_path),
        "processed_lgd_inputs": str(processed_lgd_path),
        "processed_ead_inputs": str(processed_ead_path),
        "ead_projection_sample": str(ead_projection_path),
        "release47_reconciliation_table": str(reconciliation_path),
        "ead_validation_table": str(ead_validation_path),
        "cure_calibration_table": str(
            table_directory / "phase3_cure_calibration.csv"
        ),
        "severity_calibration_table": str(
            table_directory / "phase3_severity_calibration.csv"
        ),
        **figures,
    }
    summary = {
        "run_completed_at_utc": utc_timestamp(),
        "run_signature": run_signature,
        "release": 47,
        "settings": settings.to_dict(),
        "memory_policy": "Monthly panels are read and checkpointed one quarter at a time",
        "cure_definition": {
            "event": (
                f"{settings.cure_consecutive_current_months} consecutive adjacent "
                "CURRENT months"
            ),
            "horizon_months": settings.cure_horizon_months,
            "development_label_cutoff": settings.development_end_month,
            "validation_label_cutoff": settings.validation_end_month,
            "validation_episode_starts_after": settings.development_end_month,
            "code16_treatment": "censored RPL exit",
        },
        "realized_lgd_definition": (
            "Disclosed Actual Loss divided by disclosed Zero Balance Removal UPB. "
            "Raw values outside zero to one are retained for audit and excluded from "
            "the fractional logit fit."
        ),
        "release47_actual_loss_sign": (
            "Zero Balance Removal UPB plus Net Sale Proceeds plus Delinquent Accrued "
            "Interest plus Total Expenses plus MI Recoveries plus Non-MI Recoveries"
        ),
        "terminal_code_policy": {
            "default_codes": sorted(DEFAULT_ZERO_BALANCE_CODES),
            "code15": "default disposition",
            "code16": "censored RPL exit",
        },
        "cure_model": {
            "numeric_features": list(CURE_NUMERIC_FEATURES),
            "categorical_features": list(CURE_CATEGORICAL_FEATURES),
            "development_diagnostics": cure_model.diagnostics.to_dict(),
            "out_of_time_validation": cure_metrics,
        },
        "severity_model": {
            "numeric_features": list(SEVERITY_NUMERIC_FEATURES),
            "categorical_features": list(SEVERITY_CATEGORICAL_FEATURES),
            "condition": "resolved default disposition",
            "development_diagnostics": severity_model.diagnostics.to_dict(),
            "out_of_time_validation": severity_metrics,
        },
        "lgd_adapter_policy": {
            "current": "conditional_lgd; cure is not applicable",
            "30_dpd": "(1 - cure_probability) * conditional_lgd",
            "60_dpd": "(1 - cure_probability) * conditional_lgd",
            "90_plus": (
                "(1 - cure_probability) * conditional_lgd because the Freddie row "
                "is nonterminal even when downstream staging assigns Stage 3"
            ),
            "defaulted": (
                "conditional_lgd with cure_probability set to zero and marked not "
                "applicable"
            ),
        },
        "ead": {
            "snapshot_month": settings.snapshot_month,
            "snapshot_reporting_loans": len(snapshot_inputs),
            "snapshot_defaulted_loans": int(
                snapshot_inputs["state"].eq(DEFAULTED).sum()
            ),
            "snapshot_weighted_loans": float(snapshot_inputs["sample_weight"].sum()),
            "scalar_definition": (
                "Interest-bearing UPB plus non-interest-bearing UPB plus delinquent "
                "accrued interest. Terminal defaults use disclosed Zero Balance "
                "Removal UPB as the principal basis when available."
            ),
            "projection": (
                "Contractual fixed-rate scheduled amortization with disclosed current "
                "rate and remaining term. Snapshot add-ons are held constant in the "
                "saved audit sample."
            ),
            "out_of_time_validation": ead_pooled,
            "projection_sample_loans": int(ead_projection["loan_id"].nunique())
            if len(ead_projection)
            else 0,
        },
        "adapter_schemas": {
            "snapshot_inputs": list(snapshot_inputs.columns),
            "lgd_inputs": list(lgd_inputs.columns),
            "ead_inputs": list(ead_inputs.columns),
        },
        "quarter_checkpoints": quarter_metadata,
        "source_manifest": source_manifest,
        "outputs": outputs,
    }
    _atomic_json(summary, summary_path)
    return summary

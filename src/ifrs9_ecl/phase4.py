"""Phase 4 loan staging and probability-weighted ECL integration.

The integration boundary is deliberately narrow. One row represents one loan
at the reporting date. PD curves may already be attached to that row or may be
supplied as separate one-row-per-loan tables. This keeps model estimation out
of the reporting-date calculation and prevents later performance from leaking
into staging or ECL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ProjectConfig
from .ead import project_amortizing_ead
from .ecl import calculate_ecl, validate_marginal_pd
from .pd_model import DiscreteTimeHazardModel
from .scenarios import Scenario, calculate_scenario_ecl, normalize_scenarios
from .staging import Stage, StagingPolicy, assign_stages
from .utils import utc_timestamp, write_json


PHASE4_ARTIFACT_VERSION = 1
DEFAULT_SNAPSHOT_PATH = Path("phase3/snapshot_inputs.parquet")
DEFAULT_CURRENT_PD_PATH = Path("phase2/current_pd_curves.parquet")
DEFAULT_ORIGINATION_PD_PATH = Path("phase2/origination_pd_curves.parquet")
DEFAULT_LGD_PATH = Path("phase3/lgd_inputs.parquet")
DEFAULT_EAD_PATH = Path("phase3/ead_inputs.parquet")


@dataclass(frozen=True)
class Phase4Calculation:
    """In-memory outputs from one Phase 4 calculation."""

    loan_results: pd.DataFrame
    stage_summary: pd.DataFrame
    scenario_summary: pd.DataFrame
    scenarios: tuple[Scenario, ...]


def _curve(value: Any, name: str) -> np.ndarray:
    """Parse one curve cell and validate it as unconditional marginal PD."""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} string must contain a JSON numeric array") from exc
    return validate_marginal_pd(parsed)


def _canonical_curve_table(
    frame: pd.DataFrame,
    *,
    target_col: str,
    loan_id_col: str,
) -> pd.DataFrame:
    """Return a one-row-per-loan curve table from wide or monthly-long input."""
    if loan_id_col not in frame.columns:
        raise KeyError(f"curve table is missing {loan_id_col!r}")
    aliases = [target_col, "marginal_pd", "pd_curve"]
    source = next((column for column in aliases if column in frame.columns), None)
    if source is None:
        raise KeyError(f"curve table is missing one of {aliases}")

    month_col = next(
        (column for column in ("projection_month", "month") if column in frame.columns),
        None,
    )
    if month_col is not None and frame[loan_id_col].duplicated().any():
        work = frame[[loan_id_col, month_col, source]].copy()
        work[month_col] = pd.to_numeric(work[month_col], errors="raise")
        work[source] = pd.to_numeric(work[source], errors="raise")
        work = work.sort_values([loan_id_col, month_col], kind="stable")
        duplicated_month = work.duplicated([loan_id_col, month_col])
        if duplicated_month.any():
            raise ValueError("curve table contains duplicate loan and month rows")
        result = (
            work.groupby(loan_id_col, sort=False)[source]
            .apply(lambda values: values.to_numpy(dtype="float64"))
            .rename(target_col)
            .reset_index()
        )
    else:
        if frame[loan_id_col].duplicated().any():
            raise ValueError("curve table must contain one row per loan")
        result = frame[[loan_id_col, source]].rename(columns={source: target_col}).copy()
    result[target_col] = [
        _curve(value, f"{target_col} for loan {loan_id}")
        for loan_id, value in result[[loan_id_col, target_col]].itertuples(index=False)
    ]
    return result


def _canonical_scalar_or_curve_table(
    frame: pd.DataFrame,
    *,
    target_col: str,
    aliases: Sequence[str],
    loan_id_col: str,
) -> pd.DataFrame:
    if loan_id_col not in frame.columns:
        raise KeyError(f"input table is missing {loan_id_col!r}")
    if frame[loan_id_col].duplicated().any():
        raise ValueError(f"{target_col} table must contain one row per loan")
    source = next((column for column in (target_col, *aliases) if column in frame.columns), None)
    if source is None:
        raise KeyError(f"{target_col} table is missing one of {(target_col, *aliases)}")
    return frame[[loan_id_col, source]].rename(columns={source: target_col}).copy()


def _merge_input(
    base: pd.DataFrame,
    addition: pd.DataFrame | None,
    *,
    target_col: str,
    loan_id_col: str,
) -> pd.DataFrame:
    if target_col in base.columns:
        if addition is not None:
            raise ValueError(
                f"{target_col} is present in both snapshot and a separate input table"
            )
        return base
    if addition is None:
        return base
    merged = base.merge(
        addition,
        how="left",
        on=loan_id_col,
        validate="one_to_one",
        sort=False,
    )
    if merged[target_col].isna().any():
        missing = merged.loc[merged[target_col].isna(), loan_id_col].head(10).tolist()
        raise ValueError(f"{target_col} input is missing loans: {missing}")
    return merged


def _merge_adapter_metadata(
    base: pd.DataFrame,
    source: pd.DataFrame | None,
    *,
    loan_id_col: str,
) -> pd.DataFrame:
    """Attach nonconflicting Phase 3 weights, segments, and audit columns."""
    if source is None:
        return base
    candidates = (
        "exposure_weight",
        "expansion_weight",
        "sample_weight",
        "segment",
        "fico_segment",
        "ltv_segment",
        "delinquency_segment",
        "conditional_lgd",
        "cure_probability",
        "non_interest_bearing_upb",
        "delinquent_accrued_interest",
        "other_unpaid_amounts",
        "remaining_term_months",
        "ead_interest_bearing_upb",
        "ead_non_interest_bearing_upb",
        "ead_delinquent_accrued_interest",
        "ead_balance_basis",
        "ead_component_valid",
        "current_interest_rate",
        "remaining_months_to_legal_maturity",
    )
    additions = [column for column in candidates if column in source and column not in base]
    if not additions:
        return base
    if source[loan_id_col].duplicated().any():
        raise ValueError("Phase 3 adapter metadata must contain one row per loan")
    return base.merge(
        source[[loan_id_col, *additions]],
        on=loan_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )


def assemble_phase4_inputs(
    snapshot: pd.DataFrame,
    *,
    current_pd: pd.DataFrame | None = None,
    origination_pd: pd.DataFrame | None = None,
    lgd: pd.DataFrame | None = None,
    ead: pd.DataFrame | None = None,
    loan_id_col: str = "loan_id",
) -> pd.DataFrame:
    """Attach explicit PD, LGD, and EAD adapters to a reporting-date snapshot.

    Current and origination PD inputs can be wide curve cells or monthly-long
    tables with ``month`` or ``projection_month``. LGD and EAD can be scalar
    or curve-valued cells. Input tables are joined one to one by loan ID.
    """
    if not isinstance(snapshot, pd.DataFrame):
        raise TypeError("snapshot must be a pandas DataFrame")
    if loan_id_col not in snapshot.columns:
        raise KeyError(f"snapshot is missing {loan_id_col!r}")
    if snapshot[loan_id_col].isna().any() or snapshot[loan_id_col].duplicated().any():
        raise ValueError("snapshot loan IDs must be nonmissing and unique")
    result = snapshot.copy(deep=True)

    snapshot_aliases = {
        "current_marginal_pd": ("current_pd_curve",),
        "origination_marginal_pd": ("origination_pd_curve",),
        "lgd": ("predicted_lgd", "lgd_curve"),
        "ead": ("predicted_ead", "ead_curve"),
    }
    for target, aliases in snapshot_aliases.items():
        if target not in result.columns:
            source = next((alias for alias in aliases if alias in result.columns), None)
            if source is not None:
                result = result.rename(columns={source: target})

    current_table = (
        _canonical_curve_table(
            current_pd,
            target_col="current_marginal_pd",
            loan_id_col=loan_id_col,
        )
        if current_pd is not None
        else None
    )
    origination_table = (
        _canonical_curve_table(
            origination_pd,
            target_col="origination_marginal_pd",
            loan_id_col=loan_id_col,
        )
        if origination_pd is not None
        else None
    )
    lgd_table = (
        _canonical_scalar_or_curve_table(
            lgd,
            target_col="lgd",
            aliases=("predicted_lgd", "lgd_curve"),
            loan_id_col=loan_id_col,
        )
        if lgd is not None
        else None
    )
    ead_table = (
        _canonical_scalar_or_curve_table(
            ead,
            target_col="ead",
            aliases=("predicted_ead", "ead_curve"),
            loan_id_col=loan_id_col,
        )
        if ead is not None
        else None
    )
    for target, addition in (
        ("current_marginal_pd", current_table),
        ("origination_marginal_pd", origination_table),
        ("lgd", lgd_table),
        ("ead", ead_table),
    ):
        result = _merge_input(
            result,
            addition,
            target_col=target,
            loan_id_col=loan_id_col,
        )
    result = _merge_adapter_metadata(result, lgd, loan_id_col=loan_id_col)
    result = _merge_adapter_metadata(result, ead, loan_id_col=loan_id_col)
    return result


def pd_curve_frame_from_model(
    model: DiscreteTimeHazardModel,
    features: pd.DataFrame,
    *,
    horizon_months: int,
    loan_id_col: str = "loan_id",
    output_col: str = "current_marginal_pd",
) -> pd.DataFrame:
    """Project marginal PD curves from the fitted Phase 2 hazard model."""
    if not isinstance(model, DiscreteTimeHazardModel):
        raise TypeError("model must be a DiscreteTimeHazardModel")
    if loan_id_col not in features.columns:
        raise KeyError(f"features are missing {loan_id_col!r}")
    if features[loan_id_col].isna().any() or features[loan_id_col].duplicated().any():
        raise ValueError("feature loan IDs must be nonmissing and unique")
    curves = model.predict_pd_curves(features, horizon_months)
    return pd.DataFrame(
        {
            loan_id_col: features[loan_id_col].to_numpy(copy=True),
            output_col: [row.copy() for row in curves.marginal_pd],
        }
    )


def _numeric_probability_curve_columns(inputs: pd.DataFrame) -> pd.DataFrame:
    result = inputs.copy(deep=True)
    for column in ("current_marginal_pd", "origination_marginal_pd"):
        if column not in result.columns:
            raise KeyError(f"Phase 4 input is missing {column!r}")
        result[column] = [
            _curve(value, f"{column} at row {position}")
            for position, value in enumerate(result[column])
        ]
    return result


def _derive_days_past_due(inputs: pd.DataFrame) -> pd.Series:
    if "days_past_due" in inputs.columns:
        values = pd.to_numeric(inputs["days_past_due"], errors="raise")
    else:
        status_col = next(
            (
                column
                for column in ("current_delinquency_status", "delinquency_status")
                if column in inputs.columns
            ),
            None,
        )
        if status_col is None:
            raise KeyError(
                "Phase 4 input requires days_past_due or a numeric delinquency status"
            )
        status = pd.to_numeric(inputs[status_col], errors="coerce")
        if status.isna().any():
            raise ValueError("nonnumeric delinquency status cannot be converted to DPD")
        values = status * 30
    if values.isna().any() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("days_past_due must contain nonnegative whole numbers")
    return values.astype("int64")


def _derive_default_flag(inputs: pd.DataFrame) -> pd.Series:
    if "is_default" in inputs.columns:
        values = inputs["is_default"]
        if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise TypeError("is_default must contain only boolean values")
        return values.astype(bool)
    if "state" in inputs.columns:
        return inputs["state"].astype("string").str.upper().eq("DEFAULTED")
    return pd.Series(False, index=inputs.index, dtype=bool)


def _rate_as_decimal(inputs: pd.DataFrame, default_rate: float) -> pd.Series:
    if "effective_annual_rate" in inputs.columns:
        rate = pd.to_numeric(inputs["effective_annual_rate"], errors="raise")
    else:
        percent_col = next(
            (
                column
                for column in ("current_interest_rate", "orig_original_interest_rate")
                if column in inputs.columns
            ),
            None,
        )
        if percent_col is None:
            rate = pd.Series(float(default_rate), index=inputs.index)
        else:
            rate = pd.to_numeric(inputs[percent_col], errors="raise") / 100.0
    if rate.isna().any() or not np.isfinite(rate).all() or (rate <= -1.0).any():
        raise ValueError("effective annual rates must be finite and greater than -1")
    return rate.astype("float64")


def project_phase4_ead(
    current_actual_upb: Any,
    remaining_term_months: Any,
    effective_annual_rate: Any,
    horizon_months: int,
    *,
    snapshot_ead: Any | None = None,
) -> tuple[np.ndarray, float]:
    """Project amortizing EAD and return its explicit nonprincipal add-on.

    A scalar snapshot EAD may exceed current UPB because it includes deferred
    principal or accrued amounts. That nonprincipal difference is held as a
    visible add-on while scheduled interest-bearing principal amortizes.
    """
    principal = float(current_actual_upb)
    if not np.isfinite(principal) or principal < 0.0:
        raise ValueError("current_actual_upb must be finite and nonnegative")
    supplied = principal if snapshot_ead is None else float(snapshot_ead)
    if not np.isfinite(supplied) or supplied < 0.0:
        raise ValueError("snapshot_ead must be finite and nonnegative")
    add_on = max(supplied - principal, 0.0)
    projection = project_amortizing_ead(
        principal,
        effective_annual_rate,
        remaining_term_months,
        horizon_months=horizon_months,
        non_interest_bearing_upb=add_on,
    )
    curve = projection["default_ead"].to_numpy(dtype="float64")[1:]
    return curve, float(add_on)


def _derive_ead_curve(
    row: pd.Series,
    horizon: int,
    *,
    snapshot_ead: float | None,
) -> tuple[np.ndarray, float]:
    principal_col = next(
        (column for column in ("current_actual_upb", "current_upb") if column in row.index),
        None,
    )
    term_col = next(
        (
            column
            for column in (
                "remaining_term_months",
                "remaining_months_to_maturity",
                "remaining_months_to_legal_maturity",
            )
            if column in row.index
        ),
        None,
    )
    if principal_col is None or term_col is None:
        raise KeyError(
            "Phase 4 requires ead or current UPB and remaining term for EAD projection"
        )
    term = int(row[term_col])
    return project_phase4_ead(
        row[principal_col],
        term,
        row["effective_annual_rate"],
        horizon,
        snapshot_ead=snapshot_ead,
    )


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _numeric_scalar_or_none(value: Any) -> float | None:
    if _is_missing_scalar(value) or isinstance(value, (str, bytes)):
        return None
    if np.isscalar(value):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None
    return None


def _scheduled_ead_fields_available(row: pd.Series) -> bool:
    principal_col = next(
        (column for column in ("current_actual_upb", "current_upb") if column in row.index),
        None,
    )
    term_col = next(
        (
            column
            for column in (
                "remaining_term_months",
                "remaining_months_to_maturity",
                "remaining_months_to_legal_maturity",
            )
            if column in row.index
        ),
        None,
    )
    if principal_col is None or term_col is None:
        return False
    return not _is_missing_scalar(row[principal_col]) and not _is_missing_scalar(row[term_col])


def scenarios_from_settings(settings: Sequence[Mapping[str, Any]] | None) -> tuple[Scenario, ...]:
    """Create normalized scenarios from hazard shifts or odds multipliers."""
    if settings is None:
        settings = (
            {"name": "upside", "weight": 0.20, "hazard_multiplier": 0.75},
            {"name": "base", "weight": 0.60, "hazard_multiplier": 1.00},
            {"name": "downside", "weight": 0.20, "hazard_multiplier": 1.60},
        )
    selected: list[Scenario] = []
    for item in settings:
        if "hazard_shift" in item:
            shift = float(item["hazard_shift"])
        else:
            multiplier = float(item.get("hazard_multiplier", 1.0))
            if not np.isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError("scenario hazard_multiplier must be positive and finite")
            shift = float(np.log(multiplier))
        selected.append(
            Scenario(
                name=str(item["name"]),
                weight=float(item["weight"]),
                hazard_shift=shift,
            )
        )
    return normalize_scenarios(selected)


def calculate_phase4(
    inputs: pd.DataFrame,
    *,
    relative_pd_threshold: float = 2.0,
    absolute_pd_increase: float = 0.05,
    stage2_dpd_backstop: int = 30,
    stage3_dpd_threshold: int = 90,
    scenarios: Sequence[Scenario] | None = None,
    default_effective_annual_rate: float = 0.0,
    loan_id_col: str = "loan_id",
) -> Phase4Calculation:
    """Stage loans and calculate baseline plus scenario-weighted ECL.

    Staging uses baseline current lifetime PD only. Scenario outcomes change
    conditional hazard odds for measurement, but do not use later outcomes or
    restage loans. Stage 1 uses months 1 to 12 while Stages 2 and 3 use the
    supplied lifetime horizon and immediate credit-impaired loss respectively.
    """
    if loan_id_col not in inputs.columns:
        raise KeyError(f"Phase 4 input is missing {loan_id_col!r}")
    if inputs.empty:
        raise ValueError("Phase 4 requires at least one snapshot loan")
    if inputs[loan_id_col].isna().any() or inputs[loan_id_col].duplicated().any():
        raise ValueError("Phase 4 loan IDs must be nonmissing and unique")
    if not np.isfinite(float(absolute_pd_increase)) or not 0 <= absolute_pd_increase <= 1:
        raise ValueError("absolute_pd_increase must be between zero and one")

    work = _numeric_probability_curve_columns(inputs)
    if "lgd" not in work.columns:
        raise KeyError("Phase 4 input is missing 'lgd'")
    work["days_past_due"] = _derive_days_past_due(work)
    work["is_default"] = _derive_default_flag(work)
    work["effective_annual_rate"] = _rate_as_decimal(
        work, default_effective_annual_rate
    )
    work["current_lifetime_pd"] = work["current_marginal_pd"].map(
        lambda curve: float(np.sum(curve))
    )
    work["origination_lifetime_pd"] = work["origination_marginal_pd"].map(
        lambda curve: float(np.sum(curve))
    )
    work["absolute_pd_increase"] = (
        work["current_lifetime_pd"] - work["origination_lifetime_pd"]
    )
    work["absolute_pd_sicr_triggered"] = (
        work["absolute_pd_increase"] >= float(absolute_pd_increase)
    )

    policy = StagingPolicy(
        sicr_relative_threshold=float(relative_pd_threshold),
        stage2_dpd_backstop=int(stage2_dpd_backstop),
        stage3_dpd_backstop=int(stage3_dpd_threshold),
    )
    work = assign_stages(work, policy=policy)
    absolute_only = work["stage"].eq(int(Stage.STAGE_1)) & work[
        "absolute_pd_sicr_triggered"
    ]
    work.loc[absolute_only, "stage"] = int(Stage.STAGE_2)
    work.loc[absolute_only, "staging_reason"] = "absolute_pd_sicr"

    if "exposure_weight" not in work.columns:
        weight_col = next(
            (column for column in ("expansion_weight", "sample_weight") if column in work),
            None,
        )
        work["exposure_weight"] = 1.0 if weight_col is None else work[weight_col]
    work["exposure_weight"] = pd.to_numeric(work["exposure_weight"], errors="raise")
    if (
        work["exposure_weight"].isna().any()
        or not np.isfinite(work["exposure_weight"]).all()
        or (work["exposure_weight"] < 0).any()
    ):
        raise ValueError("exposure_weight must be finite and nonnegative")

    selected_scenarios = normalize_scenarios(
        tuple(scenarios) if scenarios is not None else scenarios_from_settings(None)
    )
    scenario_columns = [f"scenario_{scenario.name}_ecl" for scenario in selected_scenarios]
    records: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        current_curve = row["current_marginal_pd"]
        supplied_ead = row["ead"] if "ead" in work.columns else None
        supplied_scalar = _numeric_scalar_or_none(supplied_ead)
        if _scheduled_ead_fields_available(row) and (
            supplied_scalar is not None or _is_missing_scalar(supplied_ead)
        ):
            ead_curve, nonprincipal_add_on = _derive_ead_curve(
                row,
                len(current_curve),
                snapshot_ead=supplied_scalar,
            )
            principal_col = (
                "current_actual_upb"
                if "current_actual_upb" in row.index
                else "current_upb"
            )
            reporting_ead = float(row[principal_col]) + nonprincipal_add_on
            ead_method = "scheduled_amortization_with_snapshot_add_on"
        elif supplied_ead is not None and not _is_missing_scalar(supplied_ead):
            ead_curve = supplied_ead
            reporting_ead = float(np.asarray(supplied_ead).reshape(-1)[0])
            nonprincipal_add_on = 0.0
            ead_method = "supplied_ead_fallback"
        else:
            raise KeyError(
                "Phase 4 requires scheduled EAD fields or a supplied EAD input"
            )
        stage = int(row["stage"])
        ead_input = reporting_ead if stage == int(Stage.STAGE_3) else ead_curve
        baseline = calculate_ecl(
            current_curve,
            row["lgd"],
            ead_input,
            stage=stage,
            effective_annual_rate=row["effective_annual_rate"],
        )
        scenario_result = calculate_scenario_ecl(
            current_curve,
            row["lgd"],
            ead_input,
            stage=stage,
            effective_annual_rate=row["effective_annual_rate"],
            scenarios=selected_scenarios,
        )
        record = {
            "baseline_ecl": baseline.total_ecl,
            "probability_weighted_ecl": scenario_result.weighted_ecl,
            "ecl_horizon_months": baseline.horizon_months,
            "ecl_method": baseline.method,
            "ead_at_reporting_date": reporting_ead,
            "ead_nonprincipal_add_on": nonprincipal_add_on,
            "ead_method": ead_method,
            "ead_source": ead_method,
        }
        record.update(
            {
                f"scenario_{outcome.scenario.name}_ecl": outcome.ecl
                for outcome in scenario_result.outcomes
            }
        )
        records.append(record)

    calculations = pd.DataFrame(records, index=work.index)
    work = pd.concat([work, calculations], axis=1)
    work["weighted_baseline_ecl"] = work["baseline_ecl"] * work["exposure_weight"]
    work["weighted_ecl"] = work["probability_weighted_ecl"] * work["exposure_weight"]
    for column in scenario_columns:
        work[f"weighted_{column}"] = work[column] * work["exposure_weight"]

    stage_summary = (
        work.groupby(["stage", "staging_reason"], sort=True, dropna=False)
        .agg(
            loan_count=(loan_id_col, "size"),
            weighted_loan_count=("exposure_weight", "sum"),
            reporting_ead=("ead_at_reporting_date", "sum"),
            baseline_ecl=("baseline_ecl", "sum"),
            probability_weighted_ecl=("probability_weighted_ecl", "sum"),
            weighted_reporting_ead=(
                "ead_at_reporting_date",
                lambda values: float(
                    np.dot(values, work.loc[values.index, "exposure_weight"])
                ),
            ),
            weighted_baseline_ecl=("weighted_baseline_ecl", "sum"),
            weighted_ecl=("weighted_ecl", "sum"),
        )
        .reset_index()
    )

    scenario_rows = []
    for scenario in selected_scenarios:
        column = f"scenario_{scenario.name}_ecl"
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "weight": scenario.weight,
                "hazard_shift": scenario.hazard_shift,
                "hazard_odds_multiplier": float(np.exp(scenario.hazard_shift)),
                "loan_count": len(work),
                "scenario_ecl": float(work[column].sum()),
                "weighted_portfolio_ecl": float(work[column].sum() * scenario.weight),
                "expansion_weighted_scenario_ecl": float(
                    work[f"weighted_{column}"].sum()
                ),
                "expansion_and_probability_weighted_ecl": float(
                    work[f"weighted_{column}"].sum() * scenario.weight
                ),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows)

    drop_curves = [
        column
        for column in ("current_marginal_pd", "origination_marginal_pd", "lgd", "ead")
        if column in work.columns
    ]
    loan_results = work.drop(columns=drop_curves)
    return Phase4Calculation(
        loan_results=loan_results,
        stage_summary=stage_summary,
        scenario_summary=scenario_summary,
        scenarios=selected_scenarios,
    )


def _resolve_path(config: ProjectConfig, value: Any, default: Path) -> Path:
    path = Path(str(value)) if value is not None else default
    if path.is_absolute():
        return path.resolve()
    return (config.path("processed_data") / path).resolve()


def _phase2_artifact_or_processed(
    config: ProjectConfig,
    *,
    configured: Any,
    artifact_name: str,
    processed_default: Path,
) -> Path:
    """Prefer the Phase 2 artifact contract, with the older processed fallback."""
    if configured is not None:
        return _resolve_path(config, configured, processed_default)
    artifact_path = config.path("artifacts") / "phase2" / artifact_name
    if artifact_path.is_file():
        return artifact_path.resolve()
    return _resolve_path(config, None, processed_default)


def _phase3_artifact_or_processed(
    config: ProjectConfig,
    *,
    configured: Any,
    artifact_name: str,
    processed_default: Path,
) -> Path:
    if configured is not None:
        return _resolve_path(config, configured, processed_default)
    artifact_path = config.path("artifacts") / "phase3" / artifact_name
    if artifact_path.is_file():
        return artifact_path.resolve()
    return _resolve_path(config, None, processed_default)


def _optional_read(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _file_signature(paths: Iterable[Path], settings: Mapping[str, Any]) -> str:
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


def _stage_figure(stage_summary: pd.DataFrame, path: Path) -> None:
    totals = stage_summary.groupby("stage", sort=True)["weighted_ecl"].sum()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.bar([f"Stage {int(stage)}" for stage in totals.index], totals.values, color="#2463a6")
    axis.set_ylabel("Probability-weighted ECL")
    axis.set_title("IFRS9-style ECL by impairment stage")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_phase4(config: ProjectConfig) -> dict[str, Any]:
    """Run Phase 4 from checkpoint inputs and persist auditable outputs."""
    if not isinstance(config, ProjectConfig):
        raise TypeError("config must be a ProjectConfig")
    config.ensure_output_directories()
    settings = config.raw.get("phase4", {})
    snapshot_path = _phase2_artifact_or_processed(
        config,
        configured=settings.get("snapshot_path"),
        artifact_name="snapshot_loans.parquet",
        processed_default=DEFAULT_SNAPSHOT_PATH,
    )
    if not snapshot_path.is_file() and settings.get("snapshot_path") is None:
        snapshot_path = _phase3_artifact_or_processed(
            config,
            configured=None,
            artifact_name="snapshot_inputs.parquet",
            processed_default=DEFAULT_SNAPSHOT_PATH,
        )
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Phase 4 snapshot input not found: {snapshot_path}")
    paths = {
        "current_pd": _phase2_artifact_or_processed(
            config,
            configured=settings.get("current_pd_path"),
            artifact_name="current_pd_curves.parquet",
            processed_default=DEFAULT_CURRENT_PD_PATH,
        ),
        "origination_pd": _phase2_artifact_or_processed(
            config,
            configured=settings.get("origination_pd_path"),
            artifact_name="origination_pd_curves.parquet",
            processed_default=DEFAULT_ORIGINATION_PD_PATH,
        ),
        "lgd": _phase3_artifact_or_processed(
            config,
            configured=settings.get("lgd_path"),
            artifact_name="lgd_inputs.parquet",
            processed_default=DEFAULT_LGD_PATH,
        ),
        "ead": _phase3_artifact_or_processed(
            config,
            configured=settings.get("ead_path"),
            artifact_name="ead_inputs.parquet",
            processed_default=DEFAULT_EAD_PATH,
        ),
    }
    input_paths = [snapshot_path, *paths.values()]
    signature_settings = {
        "ifrs9": config.raw.get("ifrs9", {}),
        "scenarios": config.raw.get("scenarios"),
        "default_effective_annual_rate": settings.get(
            "default_effective_annual_rate", 0.0
        ),
    }
    input_signature = _file_signature(input_paths, signature_settings)

    processed = config.path("processed_data") / "phase4"
    artifact = config.path("artifacts") / "phase4"
    figure_path = config.path("figures") / "phase4_ecl_by_stage.png"
    loan_path = processed / "loan_ecl.parquet"
    stage_path = artifact / "tables" / "stage_summary.csv"
    scenario_path = artifact / "tables" / "scenario_summary.csv"
    summary_path = artifact / "phase4_summary.json"
    checkpoint_outputs = (loan_path, stage_path, scenario_path, figure_path)
    if (
        bool(settings.get("reuse_completed", True))
        and summary_path.is_file()
        and all(path.is_file() for path in checkpoint_outputs)
    ):
        with summary_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            existing.get("artifact_version") == PHASE4_ARTIFACT_VERSION
            and existing.get("input_signature") == input_signature
        ):
            return existing

    snapshot = _optional_read(snapshot_path)
    assert snapshot is not None
    inputs = assemble_phase4_inputs(
        snapshot,
        current_pd=_optional_read(paths["current_pd"]),
        origination_pd=_optional_read(paths["origination_pd"]),
        lgd=_optional_read(paths["lgd"]),
        ead=_optional_read(paths["ead"]),
    )
    ifrs9 = config.raw.get("ifrs9", {})
    calculation = calculate_phase4(
        inputs,
        relative_pd_threshold=float(ifrs9.get("sicr_relative_lifetime_pd_ratio", 2.0)),
        absolute_pd_increase=float(
            ifrs9.get("sicr_absolute_lifetime_pd_increase", 0.05)
        ),
        stage2_dpd_backstop=int(ifrs9.get("stage2_dpd_backstop", 30)),
        stage3_dpd_threshold=int(ifrs9.get("stage3_dpd_threshold", 90)),
        scenarios=scenarios_from_settings(config.raw.get("scenarios")),
        default_effective_annual_rate=float(
            settings.get("default_effective_annual_rate", 0.0)
        ),
    )
    _atomic_parquet(calculation.loan_results, loan_path)
    _atomic_csv(calculation.stage_summary, stage_path)
    _atomic_csv(calculation.scenario_summary, scenario_path)
    _stage_figure(calculation.stage_summary, figure_path)

    loans = calculation.loan_results
    summary: dict[str, Any] = {
        "artifact_version": PHASE4_ARTIFACT_VERSION,
        "run_completed_at_utc": utc_timestamp(),
        "input_signature": input_signature,
        "snapshot_loans": len(loans),
        "staging_policy": {
            "relative_lifetime_pd_ratio": float(
                ifrs9.get("sicr_relative_lifetime_pd_ratio", 2.0)
            ),
            "absolute_lifetime_pd_increase": float(
                ifrs9.get("sicr_absolute_lifetime_pd_increase", 0.05)
            ),
            "stage2_dpd_backstop": int(ifrs9.get("stage2_dpd_backstop", 30)),
            "stage3_dpd_threshold": int(ifrs9.get("stage3_dpd_threshold", 90)),
        },
        "scenario_method": "constant log-odds shift to monthly conditional hazards",
        "scenario_weights_sum": float(sum(item.weight for item in calculation.scenarios)),
        "total_probability_weighted_ecl": float(loans["probability_weighted_ecl"].sum()),
        "total_expansion_weighted_ecl": float(loans["weighted_ecl"].sum()),
        "stage_counts": {
            str(int(stage)): int(count)
            for stage, count in loans["stage"].value_counts().sort_index().items()
        },
        "ead_method_counts": {
            str(method): int(count)
            for method, count in loans["ead_method"].value_counts().sort_index().items()
        },
        "inputs": {
            "snapshot": str(snapshot_path),
            **{name: str(path) if path.is_file() else None for name, path in paths.items()},
        },
        "outputs": {
            "loan_ecl": str(loan_path),
            "stage_summary": str(stage_path),
            "scenario_summary": str(scenario_path),
            "stage_figure": str(figure_path),
        },
    }
    write_json(summary, summary_path)
    return summary

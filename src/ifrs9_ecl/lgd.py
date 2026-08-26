"""Delinquency cure and realized loss severity utilities.

The functions in this module keep observed outcomes separate from modeling
treatments. In particular, realized LGD is always reported as the unmodified
Actual Loss divided by Zero Balance Removal UPB. A caller can retain, clip, or
exclude values outside the unit interval for modeling without losing the raw
ratio or its audit flags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Literal, Sequence
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .states import (
    CENSORED_DEFECT,
    CENSORED_RPL,
    CURRENT,
    DEFAULTED,
    DPD_30,
    DPD_60,
    DPD_90_PLUS,
    PAID_OFF,
    UNKNOWN,
    coerce_state,
)


DELINQUENT_STATES = (DPD_30, DPD_60, DPD_90_PLUS)
DEFAULT_CENSOR_STATES = (PAID_OFF, CENSORED_RPL, CENSORED_DEFECT, UNKNOWN)
OUT_OF_RANGE_TREATMENTS = frozenset({"retain", "clip", "exclude", "error"})


@dataclass(frozen=True)
class CureDefinition:
    """Rules used to resolve and label a delinquency episode.

    ``consecutive_current_months`` controls whether a cure must be sustained.
    ``outcome_horizon_months`` defines a cure-within-horizon binary target. An
    unresolved episode is a known non-cure only after that many adjacent months
    have been observed. Earlier panel exits and gaps remain censored.
    """

    delinquent_states: tuple[str, ...] = DELINQUENT_STATES
    cure_states: tuple[str, ...] = (CURRENT,)
    default_states: tuple[str, ...] = (DEFAULTED,)
    censor_states: tuple[str, ...] = DEFAULT_CENSOR_STATES
    consecutive_current_months: int = 1
    outcome_horizon_months: int | None = 12

    def __post_init__(self) -> None:
        if isinstance(self.consecutive_current_months, bool) or not isinstance(
            self.consecutive_current_months, Integral
        ):
            raise TypeError("consecutive_current_months must be an integer")
        if self.consecutive_current_months < 1:
            raise ValueError("consecutive_current_months must be at least one")
        if self.outcome_horizon_months is not None:
            if isinstance(self.outcome_horizon_months, bool) or not isinstance(
                self.outcome_horizon_months, Integral
            ):
                raise TypeError("outcome_horizon_months must be an integer or None")
            if self.outcome_horizon_months < 1:
                raise ValueError("outcome_horizon_months must be at least one")

        groups = {
            "delinquent_states": tuple(coerce_state(value) for value in self.delinquent_states),
            "cure_states": tuple(coerce_state(value) for value in self.cure_states),
            "default_states": tuple(coerce_state(value) for value in self.default_states),
            "censor_states": tuple(coerce_state(value) for value in self.censor_states),
        }
        for name, values in groups.items():
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate states")
            object.__setattr__(self, name, values)

        membership: dict[str, str] = {}
        for group_name, values in groups.items():
            for value in values:
                if value in membership:
                    raise ValueError(
                        f"state {value!r} appears in both {membership[value]} and {group_name}"
                    )
                membership[value] = group_name


@dataclass(frozen=True)
class EpisodeDiagnostics:
    """Audit counts produced while constructing delinquency episodes."""

    input_rows: int
    input_loans: int
    usable_loan_months: int
    identical_duplicate_rows_collapsed: int
    conflicting_loan_months_dropped: int
    gaps_encountered: int
    episodes: int
    cured_within_horizon: int
    noncures_within_horizon: int
    censored_outcomes: int
    default_events: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResult:
    """One-row-per-episode output and construction diagnostics."""

    episodes: pd.DataFrame
    diagnostics: EpisodeDiagnostics


@dataclass(frozen=True)
class CureModelDiagnostics:
    """Convergence and in-sample fit evidence for a cure model."""

    input_rows: int
    excluded_missing_target_rows: int
    fitted_rows: int
    positive_rows: int
    negative_rows: int
    weight_sum: float
    effective_sample_size: float
    weighted_cure_rate: float
    encoded_feature_count: int
    converged: bool
    iterations: int
    max_iterations: int
    log_loss: float
    brier_score: float
    roc_auc: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightedCureModel:
    """A fitted preprocessing and weighted logistic regression pipeline."""

    pipeline: Pipeline
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    diagnostics: CureModelDiagnostics

    def predict_probability(self, data: pd.DataFrame) -> np.ndarray:
        """Return predicted cure probabilities for new observations."""

        features = _prepare_feature_frame(
            data, self.numeric_features, self.categorical_features
        )
        return np.asarray(self.pipeline.predict_proba(features)[:, 1], dtype="float64")

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return the usual two-column class probability matrix."""

        features = _prepare_feature_frame(
            data, self.numeric_features, self.categorical_features
        )
        return np.asarray(self.pipeline.predict_proba(features), dtype="float64")


@dataclass(frozen=True)
class RealizedLGDObservation:
    """Raw and modeling LGD for one resolved loss record."""

    actual_loss: float | None
    zero_balance_removal_upb: float | None
    raw_lgd: float | None
    modeling_lgd: float | None
    is_gain: bool
    below_zero: bool
    above_one: bool
    outside_unit_interval: bool
    eligible_for_model: bool
    treatment: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeverityModelDiagnostics:
    """Convergence and fit evidence for a conditional fractional logit model."""

    input_rows: int
    condition_eligible_rows: int
    excluded_by_condition_rows: int
    excluded_missing_target_rows: int
    outside_unit_interval_rows: int
    excluded_outside_unit_interval_rows: int
    clipped_outside_unit_interval_rows: int
    zero_weight_rows: int
    fitted_rows: int
    weight_sum: float
    effective_sample_size: float
    encoded_feature_count: int
    converged: bool
    iterations: int
    max_iterations: int
    deviance: float
    null_deviance: float
    aic: float
    pseudo_r_squared: float
    weighted_mean_observed: float
    weighted_mean_predicted: float
    weighted_mae: float
    weighted_rmse: float
    warnings: tuple[str, ...]
    out_of_range_treatment: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionalSeverityModel:
    """A conditional LGD fractional logit model with its fitted preprocessor."""

    preprocessor: ColumnTransformer
    model_result: Any
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    diagnostics: SeverityModelDiagnostics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return conditional expected severity bounded by zero and one."""

        features = _prepare_feature_frame(
            data, self.numeric_features, self.categorical_features
        )
        encoded = np.asarray(self.preprocessor.transform(features), dtype="float64")
        design = sm.add_constant(encoded, prepend=True, has_constant="add")
        return np.asarray(self.model_result.predict(design), dtype="float64")


def _month_period(value: Any) -> pd.Period:
    if value is None or value is pd.NaT:
        raise ValueError("reporting month is missing")
    try:
        if bool(pd.isna(value)):
            raise ValueError("reporting month is missing")
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Period):
        return value.asfreq("M")
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Period(value, freq="M")
    if isinstance(value, bool):
        raise ValueError(f"invalid reporting month: {value!r}")
    if isinstance(value, Integral):
        text = str(int(value))
    elif isinstance(value, Real) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if len(text) == 6 and text.isdigit():
        year = int(text[:4])
        month = int(text[4:])
        if not 1 <= month <= 12:
            raise ValueError(f"invalid reporting month: {value!r}")
        return pd.Period(year=year, month=month, freq="M")
    try:
        return pd.Period(text, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid reporting month: {value!r}") from exc


def _parse_month_series(values: pd.Series) -> pd.Series:
    parsed: list[pd.Period] = []
    invalid: list[Any] = []
    for value in values:
        try:
            parsed.append(_month_period(value))
        except ValueError:
            parsed.append(pd.NaT)
            invalid.append(value)
    if invalid:
        raise ValueError(
            f"{len(invalid)} reporting month values are invalid, examples: {invalid[:5]}"
        )
    return pd.Series(pd.PeriodIndex(parsed, freq="M"), index=values.index)


def _empty_episode_frame(feature_cols: Sequence[str]) -> pd.DataFrame:
    columns = [
        "loan_id",
        "episode_number",
        "episode_id",
        "start_month",
        "start_state",
        "previous_state",
        "left_censored_start",
        "last_delinquent_month",
        "worst_state",
        "delinquent_months",
        "last_observed_month",
        "event_month",
        "cure_confirmation_month",
        "event_state",
        "event_type",
        "months_to_event",
        "months_observed",
        "cured",
        "outcome_observed",
        "outcome_reason",
    ]
    return pd.DataFrame(columns=[*columns, *feature_cols])


def build_delinquency_episodes(
    panel: pd.DataFrame,
    *,
    loan_id_col: str = "loan_id",
    month_col: str = "reporting_month",
    state_col: str = "state",
    definition: CureDefinition | None = None,
    feature_cols: Sequence[str] = (),
) -> EpisodeResult:
    """Construct delinquency episodes and an auditable cure target.

    Exact duplicate loan-month-state rows are collapsed. Conflicting states for
    one loan-month are dropped and create a gap. A gap while delinquent censors
    the active episode. Cure confirmation requires adjacent current months.

    ``cured`` is nullable. It is one for a confirmed cure within the horizon and
    zero for default or survival without cure through the horizon. It is missing
    when observation stops too early. With no horizon, only cure and default are
    labeled and all other exits are censored.
    """

    rules = definition or CureDefinition()
    features = tuple(feature_cols)
    if len(set(features)) != len(features):
        raise ValueError("feature_cols contains duplicates")
    required = (loan_id_col, month_col, state_col, *features)
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing episode input columns: {missing}")
    if panel[loan_id_col].isna().any():
        raise ValueError("loan identifiers must not be missing")

    work = panel.loc[:, required].copy()
    work["_month"] = _parse_month_series(work[month_col])
    work["_state"] = work[state_col].map(coerce_state)
    work["_row_order"] = np.arange(len(work), dtype="int64")

    input_rows = len(work)
    input_loans = int(work[loan_id_col].nunique())
    keys = [loan_id_col, "_month"]
    duplicate_audit = (
        work.groupby(keys, sort=False, observed=True)["_state"]
        .agg(rows="size", distinct_states="nunique")
        .reset_index()
    )
    identical = (duplicate_audit["rows"] > 1) & (
        duplicate_audit["distinct_states"] == 1
    )
    conflicting = duplicate_audit["distinct_states"] > 1
    identical_rows_collapsed = int(
        (duplicate_audit.loc[identical, "rows"] - 1).sum()
    )
    conflicting_months = int(conflicting.sum())
    conflict_keys = pd.MultiIndex.from_frame(duplicate_audit.loc[conflicting, keys])
    if len(conflict_keys):
        row_keys = pd.MultiIndex.from_frame(work[keys])
        work = work.loc[~row_keys.isin(conflict_keys)].copy()
    work = work.sort_values(
        [loan_id_col, "_month", "_row_order"], kind="stable"
    ).drop_duplicates(keys, keep="first")

    delinquent = set(rules.delinquent_states)
    cures = set(rules.cure_states)
    defaults = set(rules.default_states)
    censors = set(rules.censor_states)
    severity_rank = {state: rank for rank, state in enumerate(rules.delinquent_states)}
    rows: list[dict[str, Any]] = []
    gaps_encountered = 0

    def close_episode(
        active: dict[str, Any],
        *,
        event_type: str,
        event_state: str | None,
        event_month: pd.Period,
        confirmation_month: pd.Period | None,
        last_observed_month: pd.Period,
    ) -> None:
        horizon = rules.outcome_horizon_months
        observed_elapsed = int(last_observed_month.ordinal - active["_start_ordinal"])
        event_elapsed = int(event_month.ordinal - active["_start_ordinal"])
        confirmation_elapsed = (
            None
            if confirmation_month is None
            else int(confirmation_month.ordinal - active["_start_ordinal"])
        )

        cured_value: int | None
        outcome_reason: str
        if horizon is None:
            if event_type == "cured":
                cured_value = 1
                outcome_reason = "confirmed_cure"
            elif event_type == "defaulted":
                cured_value = 0
                outcome_reason = "default_before_cure"
            else:
                cured_value = None
                outcome_reason = f"censored_{event_type}"
        elif event_type == "cured" and confirmation_elapsed is not None and (
            confirmation_elapsed <= horizon
        ):
            cured_value = 1
            outcome_reason = "cured_within_horizon"
        elif event_type == "defaulted" and event_elapsed <= horizon:
            cured_value = 0
            outcome_reason = "defaulted_within_horizon"
        elif observed_elapsed >= horizon:
            cured_value = 0
            outcome_reason = "not_cured_within_horizon"
        else:
            cured_value = None
            outcome_reason = f"censored_before_horizon_{event_type}"

        result = {
            "loan_id": active["loan_id"],
            "episode_number": active["episode_number"],
            "episode_id": f"{active['loan_id']}:{active['episode_number']}",
            "start_month": active["start_month"],
            "start_state": active["start_state"],
            "previous_state": active["previous_state"],
            "left_censored_start": active["left_censored_start"],
            "last_delinquent_month": active["last_delinquent_month"],
            "worst_state": active["worst_state"],
            "delinquent_months": active["delinquent_months"],
            "last_observed_month": last_observed_month,
            "event_month": event_month,
            "cure_confirmation_month": confirmation_month,
            "event_state": event_state,
            "event_type": event_type,
            "months_to_event": event_elapsed,
            "months_observed": observed_elapsed + 1,
            "cured": cured_value,
            "outcome_observed": cured_value is not None,
            "outcome_reason": outcome_reason,
        }
        result.update(active["features"])
        rows.append(result)

    for loan_id, loan_rows in work.groupby(loan_id_col, sort=False, observed=True):
        records = loan_rows.to_dict("records")
        active: dict[str, Any] | None = None
        episode_number = 0
        previous_month: pd.Period | None = None
        previous_state: str | None = None

        for record in records:
            month = record["_month"]
            state = record["_state"]
            is_gap = previous_month is not None and month.ordinal - previous_month.ordinal != 1
            if is_gap:
                gaps_encountered += 1
                if active is not None:
                    close_episode(
                        active,
                        event_type="gap",
                        event_state=previous_state,
                        event_month=previous_month,
                        confirmation_month=None,
                        last_observed_month=previous_month,
                    )
                    active = None

            if active is None and state in delinquent:
                episode_number += 1
                prior_is_adjacent = previous_month is not None and not is_gap
                active = {
                    "loan_id": loan_id,
                    "episode_number": episode_number,
                    "start_month": month,
                    "_start_ordinal": month.ordinal,
                    "start_state": state,
                    "previous_state": previous_state if prior_is_adjacent else None,
                    "left_censored_start": not prior_is_adjacent,
                    "last_delinquent_month": month,
                    "worst_state": state,
                    "delinquent_months": 1,
                    "_cure_run": 0,
                    "_cure_start": None,
                    "features": {column: record[column] for column in features},
                }
            elif active is not None:
                if state in delinquent:
                    active["delinquent_months"] += 1
                    active["last_delinquent_month"] = month
                    active["_cure_run"] = 0
                    active["_cure_start"] = None
                    if severity_rank[state] > severity_rank[active["worst_state"]]:
                        active["worst_state"] = state
                elif state in cures:
                    if active["_cure_run"] == 0:
                        active["_cure_start"] = month
                    active["_cure_run"] += 1
                    if active["_cure_run"] >= rules.consecutive_current_months:
                        close_episode(
                            active,
                            event_type="cured",
                            event_state=state,
                            event_month=active["_cure_start"],
                            confirmation_month=month,
                            last_observed_month=month,
                        )
                        active = None
                elif state in defaults:
                    close_episode(
                        active,
                        event_type="defaulted",
                        event_state=state,
                        event_month=month,
                        confirmation_month=None,
                        last_observed_month=month,
                    )
                    active = None
                elif state in censors:
                    close_episode(
                        active,
                        event_type=("paid_off" if state == PAID_OFF else "censored"),
                        event_state=state,
                        event_month=month,
                        confirmation_month=None,
                        last_observed_month=month,
                    )
                    active = None
                else:
                    close_episode(
                        active,
                        event_type="other_exit",
                        event_state=state,
                        event_month=month,
                        confirmation_month=None,
                        last_observed_month=month,
                    )
                    active = None

            previous_month = month
            previous_state = state

        if active is not None and previous_month is not None:
            close_episode(
                active,
                event_type="panel_end",
                event_state=previous_state,
                event_month=previous_month,
                confirmation_month=None,
                last_observed_month=previous_month,
            )

    episodes = pd.DataFrame(rows) if rows else _empty_episode_frame(features)
    if "cured" in episodes:
        episodes["cured"] = pd.array(episodes["cured"], dtype="Int64")
        episodes["outcome_observed"] = episodes["outcome_observed"].astype("bool")
    cured_count = int((episodes["cured"] == 1).sum()) if len(episodes) else 0
    noncure_count = int((episodes["cured"] == 0).sum()) if len(episodes) else 0
    censored_count = int(episodes["cured"].isna().sum()) if len(episodes) else 0
    default_count = int((episodes["event_type"] == "defaulted").sum()) if len(episodes) else 0

    diagnostics = EpisodeDiagnostics(
        input_rows=input_rows,
        input_loans=input_loans,
        usable_loan_months=len(work),
        identical_duplicate_rows_collapsed=identical_rows_collapsed,
        conflicting_loan_months_dropped=conflicting_months,
        gaps_encountered=gaps_encountered,
        episodes=len(episodes),
        cured_within_horizon=cured_count,
        noncures_within_horizon=noncure_count,
        censored_outcomes=censored_count,
        default_events=default_count,
    )
    return EpisodeResult(episodes=episodes, diagnostics=diagnostics)


def construct_delinquency_episodes(*args: Any, **kwargs: Any) -> EpisodeResult:
    """Alias for :func:`build_delinquency_episodes`."""

    return build_delinquency_episodes(*args, **kwargs)


def _normalise_features(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(numeric_features)
    categorical = tuple(categorical_features)
    if not numeric and not categorical:
        raise ValueError("at least one numeric or categorical feature is required")
    if len(set(numeric)) != len(numeric):
        raise ValueError("numeric_features contains duplicates")
    if len(set(categorical)) != len(categorical):
        raise ValueError("categorical_features contains duplicates")
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(f"features cannot be both numeric and categorical: {overlap}")
    return numeric, categorical


def _prepare_feature_frame(
    data: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    columns = [*numeric_features, *categorical_features]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"missing model feature columns: {missing}")
    result = data.loc[:, columns].copy()
    for column in numeric_features:
        original_missing = result[column].isna()
        numeric = pd.to_numeric(result[column], errors="coerce")
        invalid = numeric.isna() & ~original_missing
        if invalid.any():
            examples = result.loc[invalid, column].head(5).tolist()
            raise ValueError(f"numeric feature {column!r} has invalid values: {examples}")
        if np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).any():
            raise ValueError(f"numeric feature {column!r} must contain finite values")
        result[column] = numeric.astype("float64")
    for column in categorical_features:
        result[column] = result[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
    return result


def _build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    drop_first_category: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        keep_empty_features=True,
                        missing_values=np.nan,
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first" if drop_first_category else None,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _coerce_binary_target(values: pd.Series, *, target_col: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() & ~values.isna()
    if invalid.any():
        examples = values.loc[invalid].head(5).tolist()
        raise ValueError(f"target {target_col!r} has invalid values: {examples}")
    observed = numeric.to_numpy(dtype="float64")
    if not np.isin(observed, [0.0, 1.0]).all():
        raise ValueError(f"target {target_col!r} must contain only zero and one")
    return observed.astype("int8")


def _sample_weights(
    data: pd.DataFrame, weight_col: str | None, *, context: str
) -> np.ndarray:
    if weight_col is None:
        return np.ones(len(data), dtype="float64")
    if weight_col not in data.columns:
        raise KeyError(f"missing {context} weight column: {weight_col!r}")
    raw = data[weight_col]
    parsed = pd.to_numeric(raw, errors="coerce")
    invalid = parsed.isna() | ~np.isfinite(parsed.to_numpy(dtype="float64", na_value=np.nan))
    if invalid.any():
        examples = raw.loc[invalid].head(5).tolist()
        raise ValueError(f"{context} weights must be finite, examples: {examples}")
    weights = parsed.to_numpy(dtype="float64")
    if (weights < 0).any():
        raise ValueError(f"{context} weights must be nonnegative")
    if not (weights > 0).any():
        raise ValueError(f"{context} weights must include at least one positive value")
    return weights


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


def fit_weighted_cure_model(
    data: pd.DataFrame,
    *,
    target_col: str = "cured",
    numeric_features: Sequence[str] = (),
    categorical_features: Sequence[str] = (),
    weight_col: str | None = None,
    regularization_strength: float = 1.0,
    max_iter: int = 1_000,
    tolerance: float = 1e-6,
    random_state: int = 0,
    raise_on_nonconvergence: bool = True,
) -> WeightedCureModel:
    """Fit a weighted logistic cure model with reusable preprocessing.

    Rows with censored, missing targets are excluded. Missing feature values are
    imputed within the fitted pipeline. Negative or nonfinite sample weights are
    rejected. By default, a convergence warning raises ``RuntimeError`` so an
    unconverged fit cannot be used silently.
    """

    numeric, categorical = _normalise_features(numeric_features, categorical_features)
    if target_col not in data.columns:
        raise KeyError(f"missing cure target column: {target_col!r}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(regularization_strength) or regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    observed_mask = data[target_col].notna()
    fitting = data.loc[observed_mask].copy()
    if fitting.empty:
        raise ValueError("no observed cure outcomes are available for fitting")
    target = _coerce_binary_target(fitting[target_col], target_col=target_col)
    if np.unique(target).size != 2:
        raise ValueError("cure target must contain both zero and one")
    weights = _sample_weights(fitting, weight_col, context="cure model")
    if float(weights[target == 0].sum()) <= 0 or float(weights[target == 1].sum()) <= 0:
        raise ValueError("cure model weights must assign positive weight to both classes")
    features = _prepare_feature_frame(fitting, numeric, categorical)

    preprocessor = _build_preprocessor(
        numeric, categorical, drop_first_category=False
    )
    classifier = LogisticRegression(
        C=float(regularization_strength),
        solver="lbfgs",
        max_iter=int(max_iter),
        tol=float(tolerance),
        random_state=random_state,
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(features, target, classifier__sample_weight=weights)
    warning_messages = tuple(str(item.message) for item in caught)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    iterations = int(np.max(classifier.n_iter_))
    converged = not convergence_warning

    probabilities = np.asarray(pipeline.predict_proba(features)[:, 1], dtype="float64")
    encoded_count = len(preprocessor.get_feature_names_out())
    diagnostics = CureModelDiagnostics(
        input_rows=len(data),
        excluded_missing_target_rows=int((~observed_mask).sum()),
        fitted_rows=len(fitting),
        positive_rows=int(target.sum()),
        negative_rows=int((target == 0).sum()),
        weight_sum=float(weights.sum()),
        effective_sample_size=_effective_sample_size(weights),
        weighted_cure_rate=float(np.average(target, weights=weights)),
        encoded_feature_count=encoded_count,
        converged=converged,
        iterations=iterations,
        max_iterations=int(max_iter),
        log_loss=float(log_loss(target, probabilities, sample_weight=weights, labels=[0, 1])),
        brier_score=float(brier_score_loss(target, probabilities, sample_weight=weights)),
        roc_auc=float(roc_auc_score(target, probabilities, sample_weight=weights)),
        warnings=warning_messages,
    )
    fitted = WeightedCureModel(
        pipeline=pipeline,
        numeric_features=numeric,
        categorical_features=categorical,
        diagnostics=diagnostics,
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            f"weighted cure model did not converge within {max_iter} iterations"
        )
    return fitted


def _optional_finite_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, received {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{field} must be finite, received {value!r}")
    return parsed


def _normalise_treatment(value: str, *, allow_retain: bool = True) -> str:
    treatment = str(value).strip().lower()
    if treatment == "raw":
        treatment = "retain"
    allowed = OUT_OF_RANGE_TREATMENTS if allow_retain else (OUT_OF_RANGE_TREATMENTS - {"retain"})
    if treatment not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"out_of_range treatment must be one of: {options}")
    return treatment


def calculate_realized_lgd(
    actual_loss: Any,
    zero_balance_removal_upb: Any,
    *,
    out_of_range: Literal["retain", "raw", "clip", "exclude", "error"] = "retain",
) -> RealizedLGDObservation:
    """Calculate raw realized LGD and an explicit modeling treatment.

    Negative Actual Loss is an economic gain and produces negative raw LGD.
    Actual Loss above removal UPB produces raw LGD above one. Neither case is
    silently changed. ``modeling_lgd`` reflects the requested treatment while
    ``raw_lgd`` and all flags always retain the source economics.
    """

    treatment = _normalise_treatment(out_of_range)
    loss = _optional_finite_float(actual_loss, field="actual_loss")
    exposure = _optional_finite_float(
        zero_balance_removal_upb, field="zero_balance_removal_upb"
    )
    if loss is None:
        return RealizedLGDObservation(
            actual_loss=None,
            zero_balance_removal_upb=exposure,
            raw_lgd=None,
            modeling_lgd=None,
            is_gain=False,
            below_zero=False,
            above_one=False,
            outside_unit_interval=False,
            eligible_for_model=False,
            treatment=treatment,
            status="actual_loss_missing",
        )
    if exposure is None:
        return RealizedLGDObservation(
            actual_loss=loss,
            zero_balance_removal_upb=None,
            raw_lgd=None,
            modeling_lgd=None,
            is_gain=loss < 0,
            below_zero=False,
            above_one=False,
            outside_unit_interval=False,
            eligible_for_model=False,
            treatment=treatment,
            status="removal_upb_missing",
        )
    if exposure <= 0:
        return RealizedLGDObservation(
            actual_loss=loss,
            zero_balance_removal_upb=exposure,
            raw_lgd=None,
            modeling_lgd=None,
            is_gain=loss < 0,
            below_zero=False,
            above_one=False,
            outside_unit_interval=False,
            eligible_for_model=False,
            treatment=treatment,
            status="nonpositive_removal_upb",
        )

    raw_lgd = loss / exposure
    below_zero = raw_lgd < 0
    above_one = raw_lgd > 1
    outside = below_zero or above_one
    if treatment == "error" and outside:
        raise ValueError(
            f"raw realized LGD {raw_lgd:.12g} is outside the unit interval"
        )
    if treatment == "clip":
        modeling_lgd: float | None = float(np.clip(raw_lgd, 0.0, 1.0))
    elif treatment == "exclude" and outside:
        modeling_lgd = None
    else:
        modeling_lgd = raw_lgd
    if below_zero:
        status = "gain_below_zero"
    elif above_one:
        status = "loss_above_one"
    else:
        status = "in_unit_interval"
    return RealizedLGDObservation(
        actual_loss=loss,
        zero_balance_removal_upb=exposure,
        raw_lgd=raw_lgd,
        modeling_lgd=modeling_lgd,
        is_gain=loss < 0,
        below_zero=below_zero,
        above_one=above_one,
        outside_unit_interval=outside,
        eligible_for_model=modeling_lgd is not None,
        treatment=treatment,
        status=status,
    )


def add_realized_lgd(
    data: pd.DataFrame,
    *,
    actual_loss_col: str = "actual_loss",
    exposure_col: str = "zero_balance_removal_upb",
    out_of_range: Literal["retain", "raw", "clip", "exclude", "error"] = "retain",
    prefix: str = "realized_lgd",
) -> pd.DataFrame:
    """Return a copy with raw realized LGD, modeling LGD, and audit flags."""

    missing = [
        column for column in (actual_loss_col, exposure_col) if column not in data.columns
    ]
    if missing:
        raise KeyError(f"missing realized LGD input columns: {missing}")
    observations = [
        calculate_realized_lgd(loss, exposure, out_of_range=out_of_range)
        for loss, exposure in zip(
            data[actual_loss_col], data[exposure_col], strict=True
        )
    ]
    result = data.copy()
    result[f"{prefix}_raw"] = [item.raw_lgd for item in observations]
    result[f"{prefix}_model"] = [item.modeling_lgd for item in observations]
    result[f"{prefix}_is_gain"] = [item.is_gain for item in observations]
    result[f"{prefix}_below_zero"] = [item.below_zero for item in observations]
    result[f"{prefix}_above_one"] = [item.above_one for item in observations]
    result[f"{prefix}_outside_unit_interval"] = [
        item.outside_unit_interval for item in observations
    ]
    result[f"{prefix}_eligible"] = [item.eligible_for_model for item in observations]
    result[f"{prefix}_status"] = [item.status for item in observations]
    return result


def compute_realized_lgd(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for :func:`add_realized_lgd` for DataFrame workflows."""

    return add_realized_lgd(*args, **kwargs)


def _weighted_error_metrics(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    absolute_error = np.abs(observed - predicted)
    squared_error = np.square(observed - predicted)
    return (
        float(np.average(absolute_error, weights=weights)),
        float(np.sqrt(np.average(squared_error, weights=weights))),
    )


def fit_conditional_severity_model(
    data: pd.DataFrame,
    *,
    target_col: str = "realized_lgd_model",
    numeric_features: Sequence[str] = (),
    categorical_features: Sequence[str] = (),
    condition_col: str | None = None,
    condition_value: Any = 1,
    weight_col: str | None = None,
    out_of_range: Literal["clip", "exclude", "error"] = "exclude",
    max_iter: int = 200,
    tolerance: float = 1e-8,
    raise_on_nonconvergence: bool = True,
) -> ConditionalSeverityModel:
    """Fit conditional LGD with a weighted binomial GLM fractional logit.

    The model is conditional because callers can restrict fitting to resolved
    defaults through ``condition_col``. Fractional targets in the closed unit
    interval are supported. Values outside that interval must be clipped,
    excluded, or rejected explicitly. Fit diagnostics include convergence,
    deviance, AIC, a deviance-based pseudo R-squared, and weighted errors.
    """

    numeric, categorical = _normalise_features(numeric_features, categorical_features)
    treatment = _normalise_treatment(out_of_range, allow_retain=False)
    if target_col not in data.columns:
        raise KeyError(f"missing severity target column: {target_col!r}")
    if condition_col is not None and condition_col not in data.columns:
        raise KeyError(f"missing severity condition column: {condition_col!r}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    if condition_col is None:
        condition_mask = pd.Series(True, index=data.index)
    else:
        condition_mask = data[condition_col].eq(condition_value).fillna(False)
    conditioned = data.loc[condition_mask].copy()
    target_missing = conditioned[target_col].isna()
    fitting = conditioned.loc[~target_missing].copy()
    if fitting.empty:
        raise ValueError("no observed conditional severity outcomes are available")
    numeric_target = pd.to_numeric(fitting[target_col], errors="coerce")
    invalid_target = numeric_target.isna() | ~np.isfinite(
        numeric_target.to_numpy(dtype="float64", na_value=np.nan)
    )
    if invalid_target.any():
        examples = fitting.loc[invalid_target, target_col].head(5).tolist()
        raise ValueError(f"severity target must be finite, examples: {examples}")
    outside = (numeric_target < 0) | (numeric_target > 1)
    outside_count = int(outside.sum())
    excluded_outside = 0
    clipped_outside = 0
    if outside_count and treatment == "error":
        examples = numeric_target.loc[outside].head(5).tolist()
        raise ValueError(f"severity target is outside the unit interval: {examples}")
    if treatment == "exclude":
        excluded_outside = outside_count
        fitting = fitting.loc[~outside].copy()
        numeric_target = numeric_target.loc[~outside]
    elif treatment == "clip":
        clipped_outside = outside_count
        numeric_target = numeric_target.clip(0.0, 1.0)
    if fitting.empty:
        raise ValueError("no severity observations remain after out-of-range treatment")

    weights = _sample_weights(fitting, weight_col, context="severity model")
    positive_weight = weights > 0
    zero_weight_rows = int((~positive_weight).sum())
    if zero_weight_rows:
        fitting = fitting.loc[positive_weight].copy()
        numeric_target = numeric_target.loc[positive_weight]
        weights = weights[positive_weight]
    target = numeric_target.to_numpy(dtype="float64")
    if len(target) < 2:
        raise ValueError("at least two positive-weight severity observations are required")
    if np.unique(target).size < 2:
        raise ValueError("severity target must contain at least two distinct values")

    features = _prepare_feature_frame(fitting, numeric, categorical)
    preprocessor = _build_preprocessor(
        numeric, categorical, drop_first_category=True
    )
    encoded = np.asarray(preprocessor.fit_transform(features), dtype="float64")
    design = sm.add_constant(encoded, prepend=True, has_constant="add")
    glm = sm.GLM(
        target,
        design,
        family=sm.families.Binomial(),
        freq_weights=weights,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model_result = glm.fit(maxiter=int(max_iter), tol=float(tolerance), disp=0)
    warning_messages = tuple(str(item.message) for item in caught)
    converged = bool(getattr(model_result, "converged", False))
    fit_history = getattr(model_result, "fit_history", {})
    iterations = int(fit_history.get("iteration", 0) or 0)
    predicted = np.asarray(model_result.predict(design), dtype="float64")
    weighted_mae, weighted_rmse = _weighted_error_metrics(target, predicted, weights)
    deviance = float(model_result.deviance)
    null_deviance = float(model_result.null_deviance)
    pseudo_r_squared = (
        float(1.0 - deviance / null_deviance)
        if np.isfinite(null_deviance) and null_deviance > 0
        else float("nan")
    )
    diagnostics = SeverityModelDiagnostics(
        input_rows=len(data),
        condition_eligible_rows=int(condition_mask.sum()),
        excluded_by_condition_rows=int((~condition_mask).sum()),
        excluded_missing_target_rows=int(target_missing.sum()),
        outside_unit_interval_rows=outside_count,
        excluded_outside_unit_interval_rows=excluded_outside,
        clipped_outside_unit_interval_rows=clipped_outside,
        zero_weight_rows=zero_weight_rows,
        fitted_rows=len(target),
        weight_sum=float(weights.sum()),
        effective_sample_size=_effective_sample_size(weights),
        encoded_feature_count=encoded.shape[1],
        converged=converged,
        iterations=iterations,
        max_iterations=int(max_iter),
        deviance=deviance,
        null_deviance=null_deviance,
        aic=float(model_result.aic),
        pseudo_r_squared=pseudo_r_squared,
        weighted_mean_observed=float(np.average(target, weights=weights)),
        weighted_mean_predicted=float(np.average(predicted, weights=weights)),
        weighted_mae=weighted_mae,
        weighted_rmse=weighted_rmse,
        warnings=warning_messages,
        out_of_range_treatment=treatment,
    )
    fitted_model = ConditionalSeverityModel(
        preprocessor=preprocessor,
        model_result=model_result,
        numeric_features=numeric,
        categorical_features=categorical,
        diagnostics=diagnostics,
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            f"conditional severity model did not converge within {max_iter} iterations"
        )
    return fitted_model


def fit_severity_model(*args: Any, **kwargs: Any) -> ConditionalSeverityModel:
    """Alias for :func:`fit_conditional_severity_model`."""

    return fit_conditional_severity_model(*args, **kwargs)


fit_cure_model = fit_weighted_cure_model
realized_lgd = calculate_realized_lgd
derive_realized_lgd = add_realized_lgd

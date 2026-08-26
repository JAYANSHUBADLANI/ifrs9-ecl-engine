"""Survival estimators and probability curve transformations.

The Kaplan-Meier estimator in this module is implemented directly from the
product-limit definition. It keeps event and censoring counts at every observed
time so the treatment of right-censored loans is visible in model diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd


ArrayLike = Sequence[float] | Sequence[int] | np.ndarray | pd.Series


@dataclass(frozen=True)
class KaplanMeierDiagnostics:
    """Counts and summary statistics for a Kaplan-Meier estimate."""

    observations: int
    zero_weight_observations: int
    weighted_observations: float
    events: int
    weighted_events: float
    censored: int
    weighted_censored: float
    event_rate: float
    censoring_rate: float
    unique_times: int
    tied_times: int
    mixed_event_censor_times: int
    last_observed_time: float
    median_survival_time: float | None
    survival_at_last_time: float

    def to_dict(self) -> dict[str, int | float | None]:
        """Return JSON-ready diagnostics."""
        return asdict(self)


@dataclass(frozen=True)
class KaplanMeierResult:
    """Kaplan-Meier curve and its censoring diagnostics."""

    curve: pd.DataFrame
    diagnostics: KaplanMeierDiagnostics

    @property
    def table(self) -> pd.DataFrame:
        """Alias for callers that refer to the event table."""
        return self.curve

    def survival_at(self, time: float) -> float:
        """Return the right-continuous survival estimate at ``time``."""
        value = float(time)
        if not np.isfinite(value):
            raise ValueError("time must be finite")
        eligible = self.curve.loc[self.curve["time"] <= value, "survival"]
        return 1.0 if eligible.empty else float(eligible.iloc[-1])


@dataclass(frozen=True)
class PDCurves:
    """Conditional hazards and the probability curves implied by them.

    ``survival`` is measured at the end of each period. ``marginal_pd`` is the
    probability of first default during that period. ``cumulative_pd`` is the
    probability of default by the end of the period.
    """

    conditional_hazard: np.ndarray
    survival: np.ndarray
    marginal_pd: np.ndarray
    cumulative_pd: np.ndarray
    axis: int

    def to_frame(self) -> pd.DataFrame:
        """Return a one-dimensional curve as a monthly DataFrame."""
        if self.conditional_hazard.ndim != 1:
            raise ValueError("to_frame requires a one-dimensional PD curve")
        return pd.DataFrame(
            {
                "month": np.arange(1, len(self.conditional_hazard) + 1),
                "conditional_hazard": self.conditional_hazard,
                "survival": self.survival,
                "marginal_pd": self.marginal_pd,
                "cumulative_pd": self.cumulative_pd,
            }
        )


def _as_one_dimensional_numeric(values: ArrayLike, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_binary_events(values: ArrayLike, *, length: int) -> np.ndarray:
    events = _as_one_dimensional_numeric(values, name="event_observed")
    if len(events) != length:
        raise ValueError("durations and event_observed must have the same length")
    if not np.isin(events, (0.0, 1.0)).all():
        raise ValueError("event_observed must contain only zero and one")
    return events.astype(bool)


def kaplan_meier(
    durations: ArrayLike,
    event_observed: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    confidence_level: float = 0.95,
) -> KaplanMeierResult:
    """Estimate a right-censored survival curve from the product limit.

    Events and censoring at the same time are both included in the risk set for
    that time. The event decrement is applied first, then both groups leave the
    risk set. Zero-weight observations are retained in diagnostics but do not
    affect the estimate.
    """
    observed_durations = _as_one_dimensional_numeric(durations, name="durations")
    if len(observed_durations) == 0:
        raise ValueError("at least one observation is required")
    if (observed_durations < 0.0).any():
        raise ValueError("durations must be nonnegative")

    events = _as_binary_events(event_observed, length=len(observed_durations))
    if sample_weight is None:
        weights = np.ones(len(observed_durations), dtype="float64")
    else:
        weights = _as_one_dimensional_numeric(sample_weight, name="sample_weight")
        if len(weights) != len(observed_durations):
            raise ValueError("sample_weight must have one value per observation")
        if (weights < 0.0).any():
            raise ValueError("sample_weight must be nonnegative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("sample_weight must have a positive total")

    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    positive = weights > 0.0
    work = pd.DataFrame(
        {
            "time": observed_durations[positive],
            "event": events[positive],
            "weight": weights[positive],
        }
    )
    work["event_weight"] = work["weight"] * work["event"].astype("float64")
    work["censor_weight"] = work["weight"] * (~work["event"]).astype("float64")
    grouped = (
        work.groupby("time", sort=True, observed=True)
        .agg(
            events=("event_weight", "sum"),
            censored=("censor_weight", "sum"),
            observations=("weight", "size"),
            event_observations=("event", "sum"),
        )
        .reset_index()
    )
    grouped["censor_observations"] = (
        grouped["observations"] - grouped["event_observations"]
    )

    z_score = NormalDist().inv_cdf(0.5 + level / 2.0)
    at_risk = float(weights.sum())
    survival = 1.0
    greenwood_sum = 0.0
    rows: list[dict[str, float]] = []
    for record in grouped.itertuples(index=False):
        event_weight = float(record.events)
        censor_weight = float(record.censored)
        if at_risk <= 0.0:
            raise AssertionError("risk set was exhausted before the final time")
        hazard = float(np.clip(event_weight / at_risk, 0.0, 1.0))
        survival *= 1.0 - hazard

        remaining_after_events = at_risk - event_weight
        if event_weight > 0.0 and remaining_after_events > 0.0:
            greenwood_sum += event_weight / (at_risk * remaining_after_events)
        variance = 0.0 if survival <= 0.0 else survival**2 * greenwood_sum
        standard_error = float(np.sqrt(max(variance, 0.0)))
        rows.append(
            {
                "time": float(record.time),
                "at_risk": at_risk,
                "events": event_weight,
                "censored": censor_weight,
                "hazard": hazard,
                "survival": float(np.clip(survival, 0.0, 1.0)),
                "cumulative_incidence": float(np.clip(1.0 - survival, 0.0, 1.0)),
                "greenwood_variance": variance,
                "standard_error": standard_error,
                "lower_confidence": float(
                    np.clip(survival - z_score * standard_error, 0.0, 1.0)
                ),
                "upper_confidence": float(
                    np.clip(survival + z_score * standard_error, 0.0, 1.0)
                ),
            }
        )
        at_risk -= event_weight + censor_weight
        if abs(at_risk) < 1e-12:
            at_risk = 0.0

    curve = pd.DataFrame.from_records(rows)
    median_rows = curve.loc[curve["survival"] <= 0.5, "time"]
    median_survival_time = (
        None if median_rows.empty else float(median_rows.iloc[0])
    )
    weighted_total = float(weights.sum())
    weighted_events = float(weights[events].sum())
    weighted_censored = float(weights[~events].sum())
    diagnostics = KaplanMeierDiagnostics(
        observations=len(observed_durations),
        zero_weight_observations=int((~positive).sum()),
        weighted_observations=weighted_total,
        events=int(events.sum()),
        weighted_events=weighted_events,
        censored=int((~events).sum()),
        weighted_censored=weighted_censored,
        event_rate=weighted_events / weighted_total,
        censoring_rate=weighted_censored / weighted_total,
        unique_times=len(grouped),
        tied_times=int((grouped["observations"] > 1).sum()),
        mixed_event_censor_times=int(
            (
                (grouped["event_observations"] > 0)
                & (grouped["censor_observations"] > 0)
            ).sum()
        ),
        last_observed_time=float(grouped["time"].iloc[-1]),
        median_survival_time=median_survival_time,
        survival_at_last_time=float(curve["survival"].iloc[-1]),
    )
    return KaplanMeierResult(curve=curve, diagnostics=diagnostics)


def estimate_kaplan_meier(
    durations: ArrayLike,
    event_observed: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    confidence_level: float = 0.95,
) -> KaplanMeierResult:
    """Named alias for ``kaplan_meier`` used by estimation pipelines."""
    return kaplan_meier(
        durations,
        event_observed,
        sample_weight=sample_weight,
        confidence_level=confidence_level,
    )


def conditional_hazards_to_pd(
    conditional_hazards: ArrayLike | np.ndarray,
    *,
    axis: int = -1,
) -> PDCurves:
    """Convert conditional event hazards into marginal and cumulative PD curves."""
    try:
        hazards = np.asarray(conditional_hazards, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("conditional_hazards must be numeric") from exc
    if hazards.ndim == 0:
        raise ValueError("conditional_hazards must have at least one dimension")
    if not np.isfinite(hazards).all():
        raise ValueError("conditional_hazards must contain only finite values")
    if ((hazards < 0.0) | (hazards > 1.0)).any():
        raise ValueError("conditional_hazards must be between zero and one")

    normalised_axis = np.core.numeric.normalize_axis_index(axis, hazards.ndim)
    survival = np.cumprod(1.0 - hazards, axis=normalised_axis)
    prior_survival = np.roll(survival, shift=1, axis=normalised_axis)
    first = [slice(None)] * hazards.ndim
    first[normalised_axis] = 0
    if hazards.shape[normalised_axis] > 0:
        prior_survival[tuple(first)] = 1.0
    marginal_pd = prior_survival * hazards
    cumulative_pd = 1.0 - survival
    return PDCurves(
        conditional_hazard=hazards.copy(),
        survival=survival,
        marginal_pd=marginal_pd,
        cumulative_pd=cumulative_pd,
        axis=normalised_axis,
    )


def hazards_to_pd_curves(
    conditional_hazards: ArrayLike | np.ndarray,
    *,
    axis: int = -1,
) -> PDCurves:
    """Readable alias for ``conditional_hazards_to_pd``."""
    return conditional_hazards_to_pd(conditional_hazards, axis=axis)

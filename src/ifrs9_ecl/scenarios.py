"""Probability-weighted macroeconomic scenario utilities.

Scenario shifts are applied to the log odds of monthly conditional default
hazards. The adjusted hazards are then converted back to unconditional
marginal PDs. This preserves a valid survival curve and avoids multiplying
unconditional marginal PDs independently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ecl import ECLResult, calculate_ecl, validate_marginal_pd
from .staging import Stage, normalize_stage


@dataclass(frozen=True)
class Scenario:
    """One macro scenario and its log-odds conditional-hazard shift."""

    name: str
    weight: float
    hazard_shift: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be a nonempty string")
        weight = _finite_number(self.weight, "scenario weight")
        if weight < 0.0:
            raise ValueError("scenario weight must be nonnegative")
        _finite_number(self.hazard_shift, "scenario hazard_shift")


MacroScenario = Scenario


@dataclass(frozen=True)
class ScenarioOutcome:
    """ECL result under one normalized scenario."""

    scenario: Scenario
    marginal_pd: np.ndarray
    ecl_result: ECLResult

    @property
    def ecl(self) -> float:
        return self.ecl_result.total_ecl

    @property
    def weighted_ecl(self) -> float:
        return self.scenario.weight * self.ecl


@dataclass(frozen=True)
class ScenarioECLResult:
    """Scenario-level outcomes and their probability-weighted ECL."""

    outcomes: tuple[ScenarioOutcome, ...]
    weighted_ecl: float

    def to_frame(self) -> pd.DataFrame:
        """Return one auditable row per scenario."""
        return pd.DataFrame(
            {
                "scenario": [outcome.scenario.name for outcome in self.outcomes],
                "weight": [outcome.scenario.weight for outcome in self.outcomes],
                "hazard_shift": [outcome.scenario.hazard_shift for outcome in self.outcomes],
                "ecl": [outcome.ecl for outcome in self.outcomes],
                "weighted_ecl": [outcome.weighted_ecl for outcome in self.outcomes],
            }
        )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="base", weight=0.60, hazard_shift=0.00),
    Scenario(name="upside", weight=0.20, hazard_shift=-0.35),
    Scenario(name="downside", weight=0.20, hazard_shift=0.50),
)


def normalize_scenarios(scenarios: Sequence[Scenario]) -> tuple[Scenario, ...]:
    """Validate scenarios and normalize nonnegative weights to sum to one."""
    selected = tuple(scenarios)
    if not selected:
        raise ValueError("at least one scenario is required")
    if not all(isinstance(scenario, Scenario) for scenario in selected):
        raise TypeError("scenarios must contain Scenario instances")
    names = [scenario.name.strip().lower() for scenario in selected]
    if len(set(names)) != len(names):
        raise ValueError("scenario names must be unique")
    total_weight = float(sum(scenario.weight for scenario in selected))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("scenario weights must have a positive finite sum")
    return tuple(
        replace(scenario, weight=float(scenario.weight) / total_weight) for scenario in selected
    )


def normalize_scenario_weights(scenarios: Sequence[Scenario]) -> tuple[Scenario, ...]:
    """Alias with an explicit name for :func:`normalize_scenarios`."""
    return normalize_scenarios(scenarios)


def conditional_hazards_from_marginal(marginal_pd: Any) -> np.ndarray:
    """Convert unconditional marginal default PDs to conditional hazards."""
    curve = validate_marginal_pd(marginal_pd)
    hazards = np.zeros_like(curve)
    survival = 1.0
    tolerance = 1e-12
    for index, probability in enumerate(curve):
        if survival <= tolerance:
            if probability > tolerance:
                raise ValueError("marginal_pd assigns default after survival reaches zero")
            hazards[index] = 0.0
            continue
        hazard = probability / survival
        if hazard > 1.0 + tolerance:
            raise ValueError("marginal_pd is inconsistent with remaining survival")
        hazards[index] = float(np.clip(hazard, 0.0, 1.0))
        survival = max(0.0, survival - probability)
    return hazards


def marginal_pd_from_hazards(hazards: Any) -> np.ndarray:
    """Convert monthly conditional hazards to unconditional marginal PDs."""
    try:
        if isinstance(hazards, np.ndarray):
            values = np.asarray(hazards, dtype="float64")
        else:
            values = np.asarray(list(hazards), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise TypeError("hazards must be a numeric sequence") from exc
    if values.ndim != 1 or not len(values):
        raise ValueError("hazards must be a nonempty one-dimensional sequence")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("hazards must contain finite probabilities between 0 and 1")

    marginal = np.zeros_like(values)
    survival = 1.0
    for index, hazard in enumerate(values):
        marginal[index] = survival * hazard
        survival *= 1.0 - hazard
    return marginal


def apply_hazard_shift(marginal_pd: Any, hazard_shift: float) -> np.ndarray:
    """Apply a constant log-odds shift to all conditional monthly hazards."""
    shift = _finite_number(hazard_shift, "hazard_shift")
    hazards = conditional_hazards_from_marginal(marginal_pd)
    shifted = np.empty_like(hazards)
    for index, hazard in enumerate(hazards):
        if hazard <= 0.0:
            shifted[index] = 0.0
        elif hazard >= 1.0:
            shifted[index] = 1.0
        else:
            log_odds = np.log(hazard) - np.log1p(-hazard)
            shifted_log_odds = log_odds + shift
            if shifted_log_odds >= 0.0:
                shifted[index] = 1.0 / (1.0 + np.exp(-shifted_log_odds))
            else:
                exponential = np.exp(shifted_log_odds)
                shifted[index] = exponential / (1.0 + exponential)
    return marginal_pd_from_hazards(shifted)


def calculate_scenario_ecl(
    baseline_marginal_pd: Any,
    lgd: Any,
    ead: Any,
    *,
    stage: Any,
    effective_annual_rate: float = 0.0,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    stage3_cash_shortfall: float | None = None,
    stage3_cash_shortfall_month: int = 0,
) -> ScenarioECLResult:
    """Calculate normalized probability-weighted ECL across macro scenarios."""
    normalized = normalize_scenarios(scenarios)
    selected_stage = normalize_stage(stage)
    outcomes: list[ScenarioOutcome] = []
    for scenario in normalized:
        if selected_stage is Stage.STAGE_3:
            shifted_curve = np.asarray([1.0])
        else:
            shifted_curve = apply_hazard_shift(baseline_marginal_pd, scenario.hazard_shift)
        ecl_result = calculate_ecl(
            shifted_curve,
            lgd,
            ead,
            stage=selected_stage,
            effective_annual_rate=effective_annual_rate,
            stage3_cash_shortfall=stage3_cash_shortfall,
            stage3_cash_shortfall_month=stage3_cash_shortfall_month,
        )
        outcomes.append(
            ScenarioOutcome(
                scenario=scenario,
                marginal_pd=shifted_curve,
                ecl_result=ecl_result,
            )
        )
    weighted_ecl = float(sum(outcome.weighted_ecl for outcome in outcomes))
    return ScenarioECLResult(outcomes=tuple(outcomes), weighted_ecl=weighted_ecl)

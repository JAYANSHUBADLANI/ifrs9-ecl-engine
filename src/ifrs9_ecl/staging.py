"""Transparent IFRS9-style staging rules.

The rules in this module are modeling choices for a research engine. They are
not a complete accounting policy. Stage 3 takes precedence over Stage 2, and
the delinquency backstops take precedence over the modeled PD comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


class Stage(IntEnum):
    """IFRS9 impairment stage."""

    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3


@dataclass(frozen=True)
class StagingPolicy:
    """Parameters for the quantitative staging policy.

    ``sicr_relative_threshold`` compares the current lifetime PD with the
    lifetime PD estimated at origination. For example, 2.0 means that a
    doubling of lifetime PD is a significant increase in credit risk.
    """

    sicr_relative_threshold: float = 2.0
    stage2_dpd_backstop: int = 30
    stage3_dpd_backstop: int = 90

    def __post_init__(self) -> None:
        threshold = _finite_number(self.sicr_relative_threshold, "sicr_relative_threshold")
        if threshold <= 1.0:
            raise ValueError("sicr_relative_threshold must be greater than 1")
        stage2_dpd = _nonnegative_integer(self.stage2_dpd_backstop, "stage2_dpd_backstop")
        stage3_dpd = _nonnegative_integer(self.stage3_dpd_backstop, "stage3_dpd_backstop")
        if stage2_dpd >= stage3_dpd:
            raise ValueError("stage2_dpd_backstop must be below stage3_dpd_backstop")


@dataclass(frozen=True)
class StagingDecision:
    """One staging result with an auditable primary trigger."""

    stage: Stage
    reason: str
    lifetime_pd_ratio: float
    pd_sicr_triggered: bool
    delinquency_backstop_triggered: bool


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative_integer(value: object, name: str) -> int:
    parsed = _finite_number(value, name)
    if parsed < 0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(parsed)


def _probability(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return parsed


def assess_stage(
    current_lifetime_pd: float,
    origination_lifetime_pd: float,
    days_past_due: int = 0,
    is_default: bool = False,
    *,
    policy: StagingPolicy | None = None,
) -> StagingDecision:
    """Assess one exposure using relative PD and delinquency backstops.

    A positive current PD divided by a zero origination PD is treated as an
    infinite relative increase. If both values are zero, the ratio is one.
    ``is_default`` must be a boolean so that missing or coded values cannot be
    interpreted as true by accident.
    """
    if policy is not None and not isinstance(policy, StagingPolicy):
        raise TypeError("policy must be a StagingPolicy")
    selected_policy = policy if policy is not None else StagingPolicy()
    current_pd = _probability(current_lifetime_pd, "current_lifetime_pd")
    origination_pd = _probability(origination_lifetime_pd, "origination_lifetime_pd")
    dpd = _nonnegative_integer(days_past_due, "days_past_due")
    if not isinstance(is_default, (bool, np.bool_)):
        raise TypeError("is_default must be boolean")

    if origination_pd == 0.0:
        pd_ratio = 1.0 if current_pd == 0.0 else float("inf")
    else:
        pd_ratio = current_pd / origination_pd

    pd_sicr = pd_ratio >= selected_policy.sicr_relative_threshold
    stage2_backstop = dpd >= selected_policy.stage2_dpd_backstop

    if bool(is_default):
        return StagingDecision(
            stage=Stage.STAGE_3,
            reason="default_flag",
            lifetime_pd_ratio=pd_ratio,
            pd_sicr_triggered=pd_sicr,
            delinquency_backstop_triggered=stage2_backstop,
        )
    if dpd >= selected_policy.stage3_dpd_backstop:
        return StagingDecision(
            stage=Stage.STAGE_3,
            reason=f"{int(selected_policy.stage3_dpd_backstop)}_dpd_backstop",
            lifetime_pd_ratio=pd_ratio,
            pd_sicr_triggered=pd_sicr,
            delinquency_backstop_triggered=True,
        )
    if stage2_backstop:
        return StagingDecision(
            stage=Stage.STAGE_2,
            reason=f"{int(selected_policy.stage2_dpd_backstop)}_dpd_backstop",
            lifetime_pd_ratio=pd_ratio,
            pd_sicr_triggered=pd_sicr,
            delinquency_backstop_triggered=True,
        )
    if pd_sicr:
        return StagingDecision(
            stage=Stage.STAGE_2,
            reason="relative_pd_sicr",
            lifetime_pd_ratio=pd_ratio,
            pd_sicr_triggered=True,
            delinquency_backstop_triggered=False,
        )
    return StagingDecision(
        stage=Stage.STAGE_1,
        reason="performing",
        lifetime_pd_ratio=pd_ratio,
        pd_sicr_triggered=False,
        delinquency_backstop_triggered=False,
    )


def assign_stage(
    current_lifetime_pd: float,
    origination_lifetime_pd: float,
    days_past_due: int = 0,
    is_default: bool = False,
    *,
    policy: StagingPolicy | None = None,
) -> int:
    """Return the integer impairment stage for one exposure."""
    return int(
        assess_stage(
            current_lifetime_pd=current_lifetime_pd,
            origination_lifetime_pd=origination_lifetime_pd,
            days_past_due=days_past_due,
            is_default=is_default,
            policy=policy,
        ).stage
    )


def assign_stages(
    exposures: pd.DataFrame,
    *,
    current_pd_col: str = "current_lifetime_pd",
    origination_pd_col: str = "origination_lifetime_pd",
    days_past_due_col: str = "days_past_due",
    default_col: str = "is_default",
    stage_col: str = "stage",
    reason_col: str = "staging_reason",
    pd_ratio_col: str = "lifetime_pd_ratio",
    policy: StagingPolicy | None = None,
) -> pd.DataFrame:
    """Return a copy of an exposure frame with staging audit columns.

    Column names are configurable so the staging logic is independent of a
    particular upstream panel schema.
    """
    required = [current_pd_col, origination_pd_col, days_past_due_col, default_col]
    missing = [column for column in required if column not in exposures.columns]
    if missing:
        raise KeyError(f"missing staging input columns: {missing}")
    output_names = [stage_col, reason_col, pd_ratio_col]
    if len(set(output_names)) != len(output_names):
        raise ValueError("staging output column names must be distinct")

    result = exposures.copy(deep=True)
    decisions: list[StagingDecision] = []
    for row in result.loc[:, required].itertuples(index=False, name=None):
        decisions.append(
            assess_stage(
                current_lifetime_pd=row[0],
                origination_lifetime_pd=row[1],
                days_past_due=row[2],
                is_default=row[3],
                policy=policy,
            )
        )

    result[stage_col] = pd.Series(
        [int(decision.stage) for decision in decisions],
        index=result.index,
        dtype="int8",
    )
    result[reason_col] = pd.Series(
        [decision.reason for decision in decisions], index=result.index, dtype="object"
    )
    result[pd_ratio_col] = pd.Series(
        [decision.lifetime_pd_ratio for decision in decisions],
        index=result.index,
        dtype="float64",
    )
    return result


def normalize_stage(value: Any) -> Stage:
    """Coerce an integer or common stage label to :class:`Stage`."""
    if isinstance(value, Stage):
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid impairment stage: {value!r}")
    if isinstance(value, Real) and float(value).is_integer():
        try:
            return Stage(int(value))
        except ValueError:
            pass
    if isinstance(value, str):
        label = value.strip().lower().replace("_", "").replace(" ", "")
        aliases = {
            "1": Stage.STAGE_1,
            "stage1": Stage.STAGE_1,
            "2": Stage.STAGE_2,
            "stage2": Stage.STAGE_2,
            "3": Stage.STAGE_3,
            "stage3": Stage.STAGE_3,
        }
        if label in aliases:
            return aliases[label]
    raise ValueError(f"invalid impairment stage: {value!r}")

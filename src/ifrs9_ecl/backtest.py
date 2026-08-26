"""Provision backtesting against subsequently realized discounted losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSummary:
    """Aggregate provision adequacy statistics."""

    observation_count: int
    total_provision: float
    total_discounted_realized_loss: float
    provision_gap: float
    coverage_ratio: float
    mean_gap: float
    mean_absolute_error: float
    root_mean_squared_error: float
    outcome: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """Overall, segment, and observation-level backtest outputs."""

    overall: BacktestSummary
    by_segment: pd.DataFrame
    detail: pd.DataFrame
    segment_columns: tuple[str, ...]


SUMMARY_COLUMNS = (
    "observation_count",
    "total_provision",
    "total_discounted_realized_loss",
    "provision_gap",
    "coverage_ratio",
    "mean_gap",
    "mean_absolute_error",
    "root_mean_squared_error",
    "outcome",
)


def _numeric_array(values: object, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be numeric")
    if isinstance(values, Real) and not isinstance(values, (bool, np.bool_)):
        array = np.asarray([values], dtype="float64")
    else:
        try:
            array = np.asarray(list(values), dtype="float64")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a numeric scalar or sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _broadcast(values: object, length: int, name: str) -> np.ndarray:
    array = _numeric_array(values, name)
    if len(array) == 1 and length != 1:
        array = np.repeat(array, length)
    if len(array) != length:
        raise ValueError(f"{name} must be scalar or have {length} values")
    return array


def discount_realized_losses(
    realized_losses: object,
    months_to_loss: object,
    effective_annual_rates: object = 0.0,
) -> np.ndarray:
    """Discount signed realized losses to the provision reporting date.

    Rates are effective annual rates. Losses may be negative when recoveries
    or gains exceed loss components. Months must be nonnegative, but they may
    be fractional if timing is measured more precisely than calendar months.
    """
    losses = _numeric_array(realized_losses, "realized_losses")
    months = _broadcast(months_to_loss, len(losses), "months_to_loss")
    rates = _broadcast(effective_annual_rates, len(losses), "effective_annual_rates")
    if (months < 0.0).any():
        raise ValueError("months_to_loss must be nonnegative")
    if (rates <= -1.0).any():
        raise ValueError("effective_annual_rates must be greater than -1")
    return losses * np.power(1.0 + rates, -months / 12.0)


def _coverage_ratio(provision: float, realized_loss: float) -> float:
    if realized_loss != 0.0:
        return provision / realized_loss
    if provision > 0.0:
        return float("inf")
    if provision < 0.0:
        return float("-inf")
    return float("nan")


def _summary(
    provision: np.ndarray,
    discounted_loss: np.ndarray,
    *,
    match_tolerance: float,
) -> BacktestSummary:
    errors = provision - discounted_loss
    total_provision = float(provision.sum())
    total_loss = float(discounted_loss.sum())
    gap = total_provision - total_loss
    if gap > match_tolerance:
        outcome = "over_provisioned"
    elif gap < -match_tolerance:
        outcome = "under_provisioned"
    else:
        outcome = "matched"
    if len(errors):
        mean_gap = float(errors.mean())
        mean_absolute_error = float(np.abs(errors).mean())
        root_mean_squared_error = float(np.sqrt(np.square(errors).mean()))
    else:
        mean_gap = float("nan")
        mean_absolute_error = float("nan")
        root_mean_squared_error = float("nan")
    return BacktestSummary(
        observation_count=len(errors),
        total_provision=total_provision,
        total_discounted_realized_loss=total_loss,
        provision_gap=gap,
        coverage_ratio=_coverage_ratio(total_provision, total_loss),
        mean_gap=mean_gap,
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=root_mean_squared_error,
        outcome=outcome,
    )


def _calendar_months_between(
    reporting_dates: pd.Series, realization_dates: pd.Series
) -> np.ndarray:
    reporting = pd.to_datetime(reporting_dates, errors="coerce")
    realization = pd.to_datetime(realization_dates, errors="coerce")
    if reporting.isna().any() or realization.isna().any():
        raise ValueError("reporting and realized loss dates must be valid and nonmissing")
    reporting_month = reporting.dt.to_period("M")
    realization_month = realization.dt.to_period("M")
    return (realization_month.array.asi8 - reporting_month.array.asi8).astype("float64")


def _segment_columns(segment_cols: str | Sequence[str] | None) -> tuple[str, ...]:
    if segment_cols is None:
        return ()
    if isinstance(segment_cols, str):
        selected = (segment_cols,)
    else:
        selected = tuple(segment_cols)
    if not selected or not all(isinstance(column, str) and column for column in selected):
        raise ValueError("segment_cols must contain nonempty column names")
    if len(set(selected)) != len(selected):
        raise ValueError("segment_cols contains duplicate columns")
    return selected


def backtest_provisions(
    observations: pd.DataFrame,
    *,
    provision_col: str = "provision",
    realized_loss_col: str = "realized_loss",
    segment_cols: str | Sequence[str] | None = None,
    discounted_loss_col: str | None = None,
    months_to_loss_col: str | None = None,
    effective_rate_col: str | None = None,
    effective_annual_rate: float = 0.0,
    reporting_date_col: str | None = None,
    realized_loss_date_col: str | None = None,
    match_tolerance: float = 1e-8,
) -> BacktestResult:
    """Compare modeled provisions with later realized losses.

    A pre-discounted loss column takes precedence when supplied. Otherwise,
    timing comes from ``months_to_loss_col`` or from the difference between the
    two optional date columns. With no timing input, losses are assumed to be
    measured at the provision date. Rates can be supplied per observation or
    as one scalar. Segment summaries always reconcile to the overall totals.
    """
    segments = _segment_columns(segment_cols)
    required = [provision_col, *segments]
    if discounted_loss_col is not None:
        required.append(discounted_loss_col)
    else:
        required.append(realized_loss_col)
        if months_to_loss_col is not None:
            required.append(months_to_loss_col)
        if (reporting_date_col is None) != (realized_loss_date_col is None):
            raise ValueError("both reporting and realized loss date columns are required")
        if reporting_date_col is not None and realized_loss_date_col is not None:
            required.extend([reporting_date_col, realized_loss_date_col])
        if effective_rate_col is not None:
            required.append(effective_rate_col)
    missing = [column for column in required if column not in observations.columns]
    if missing:
        raise KeyError(f"missing backtest input columns: {missing}")

    if isinstance(match_tolerance, (bool, np.bool_)) or not isinstance(match_tolerance, Real):
        raise TypeError("match_tolerance must be a real number")
    tolerance = float(match_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("match_tolerance must be finite and nonnegative")

    detail = observations.copy(deep=True)
    provision = _numeric_array(detail[provision_col], provision_col)
    if discounted_loss_col is not None:
        discounted_loss = _numeric_array(detail[discounted_loss_col], discounted_loss_col)
        factors = np.full(len(detail), np.nan, dtype="float64")
    else:
        realized_loss = _numeric_array(detail[realized_loss_col], realized_loss_col)
        if months_to_loss_col is not None:
            months = _numeric_array(detail[months_to_loss_col], months_to_loss_col)
        elif reporting_date_col is not None and realized_loss_date_col is not None:
            months = _calendar_months_between(
                detail[reporting_date_col], detail[realized_loss_date_col]
            )
        else:
            months = np.zeros(len(detail), dtype="float64")
        rates: object = (
            detail[effective_rate_col] if effective_rate_col is not None else effective_annual_rate
        )
        discounted_loss = discount_realized_losses(realized_loss, months, rates)
        rate_array = _broadcast(rates, len(detail), "effective_annual_rates")
        factors = np.power(1.0 + rate_array, -months / 12.0)

    if len(provision) != len(discounted_loss):
        raise AssertionError("validated backtest arrays have inconsistent lengths")
    detail["discount_factor"] = factors
    detail["discounted_realized_loss"] = discounted_loss
    detail["provision_gap"] = provision - discounted_loss

    overall = _summary(provision, discounted_loss, match_tolerance=tolerance)
    segment_rows: list[dict[str, object]] = []
    if segments:
        group_key: str | list[str] = segments[0] if len(segments) == 1 else list(segments)
        for key, group in detail.groupby(group_key, dropna=False, sort=True):
            keys = (key,) if len(segments) == 1 else tuple(key)
            summary = _summary(
                group[provision_col].to_numpy(dtype="float64"),
                group["discounted_realized_loss"].to_numpy(dtype="float64"),
                match_tolerance=tolerance,
            )
            row: dict[str, object] = dict(zip(segments, keys, strict=True))
            row.update(summary.to_dict())
            segment_rows.append(row)
    by_segment = pd.DataFrame(segment_rows, columns=[*segments, *SUMMARY_COLUMNS])

    return BacktestResult(
        overall=overall,
        by_segment=by_segment,
        detail=detail,
        segment_columns=segments,
    )


def summarize_backtest(*args: object, **kwargs: object) -> BacktestResult:
    """Alias for :func:`backtest_provisions`."""
    return backtest_provisions(*args, **kwargs)

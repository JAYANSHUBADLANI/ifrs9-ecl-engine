"""Observed month-to-month delinquency transition estimation.

Only records in adjacent calendar months form transitions. Gaps are reported and
skipped. Duplicate loan-month records are audited before estimation, and conflicting
duplicates are removed because choosing one state would create an unsupported path.
Terminal states are not made absorbing by adding synthetic self transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .states import STATE_ORDER, UNKNOWN, coerce_state


@dataclass(frozen=True)
class TransitionDiagnostics:
    """Audit counts for the observations accepted and skipped by the estimator."""

    input_rows: int
    input_loans: int
    unique_loan_months: int
    usable_loan_months: int
    duplicate_rows: int
    duplicate_loan_months: int
    identical_duplicate_loan_months: int
    identical_duplicate_rows_collapsed: int
    conflicting_duplicate_loan_months: int
    conflicting_rows_dropped: int
    loans_with_duplicates: int
    candidate_pairs: int
    adjacent_pairs: int
    gap_pairs_skipped: int
    loans_with_gaps: int
    largest_gap_months: int
    unknown_state_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionResult:
    """Transition counts, row probabilities, long-form output, and diagnostics."""

    counts: pd.DataFrame
    probabilities: pd.DataFrame
    long: pd.DataFrame
    diagnostics: TransitionDiagnostics


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


def _parse_months(values: pd.Series) -> pd.Series:
    parsed: list[pd.Period] = []
    invalid: list[Any] = []
    for value in values:
        try:
            parsed.append(_month_period(value))
        except ValueError:
            parsed.append(pd.NaT)
            invalid.append(value)
    if invalid:
        examples = invalid[:5]
        raise ValueError(
            f"{len(invalid)} reporting month values are invalid, examples: {examples}"
        )
    return pd.Series(pd.PeriodIndex(parsed, freq="M"), index=values.index)


def _validate_state_order(state_order: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(coerce_state(state) for state in state_order)
    if len(set(normalised)) != len(normalised):
        raise ValueError("state_order contains duplicate states")
    return normalised


def estimate_transitions(
    panel: pd.DataFrame,
    loan_id_col: str = "loan_id",
    month_col: str = "reporting_month",
    state_col: str = "state",
    state_order: Sequence[str] = STATE_ORDER,
) -> TransitionResult:
    """Estimate empirical transitions from observed adjacent monthly records.

    Exact duplicate loan-month-state records are collapsed. If one loan-month has
    conflicting states, every record for that month is excluded. The resulting gap is
    then treated like any other nonadjacent observation and cannot form a transition.

    Probability rows with no observed outgoing transition remain ``NaN``. In particular,
    this function does not set terminal-state self transitions to one.
    """
    required = (loan_id_col, month_col, state_col)
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"missing transition input columns: {missing}")
    if panel[loan_id_col].isna().any():
        raise ValueError("loan identifiers must not be missing")

    order = _validate_state_order(state_order)
    work = panel.loc[:, required].copy()
    work["_month"] = _parse_months(work[month_col])
    work["_state"] = work[state_col].map(coerce_state)
    states_missing_from_order = sorted(set(work["_state"]) - set(order))
    if states_missing_from_order:
        raise ValueError(
            f"state_order omits observed states: {states_missing_from_order}"
        )
    work["_row_order"] = np.arange(len(work), dtype="int64")

    input_rows = len(work)
    input_loans = int(work[loan_id_col].nunique())
    unknown_state_rows = int((work["_state"] == UNKNOWN).sum())

    keys = [loan_id_col, "_month"]
    duplicate_audit = (
        work.groupby(keys, sort=False, observed=True)["_state"]
        .agg(rows="size", distinct_states="nunique")
        .reset_index()
    )
    duplicate_months = duplicate_audit["rows"] > 1
    conflicting_months = duplicate_audit["distinct_states"] > 1
    identical_months = duplicate_months & ~conflicting_months

    duplicate_rows = int(
        (duplicate_audit.loc[duplicate_months, "rows"] - 1).sum()
    )
    duplicate_loan_months = int(duplicate_months.sum())
    identical_duplicate_loan_months = int(identical_months.sum())
    identical_duplicate_rows_collapsed = int(
        (duplicate_audit.loc[identical_months, "rows"] - 1).sum()
    )
    conflicting_duplicate_loan_months = int(conflicting_months.sum())
    conflicting_rows_dropped = int(
        duplicate_audit.loc[conflicting_months, "rows"].sum()
    )
    loans_with_duplicates = int(
        duplicate_audit.loc[duplicate_months, loan_id_col].nunique()
    )

    conflict_keys = pd.MultiIndex.from_frame(
        duplicate_audit.loc[conflicting_months, keys]
    )
    row_keys = pd.MultiIndex.from_frame(work[keys])
    if len(conflict_keys):
        work = work.loc[~row_keys.isin(conflict_keys)].copy()
    work = work.sort_values(
        [loan_id_col, "_month", "_row_order"], kind="stable"
    ).drop_duplicates(keys, keep="first")

    unique_loan_months = len(duplicate_audit)
    usable_loan_months = len(work)
    work["_month_ordinal"] = work["_month"].array.asi8
    grouped = work.groupby(loan_id_col, sort=False, observed=True)
    work["_previous_ordinal"] = grouped["_month_ordinal"].shift()
    work["_from_state"] = grouped["_state"].shift()
    work["_month_delta"] = work["_month_ordinal"] - work["_previous_ordinal"]

    candidates = work[work["_previous_ordinal"].notna()].copy()
    adjacent = candidates[candidates["_month_delta"] == 1].copy()
    gaps = candidates[candidates["_month_delta"] > 1].copy()
    unexpected = candidates[candidates["_month_delta"] <= 0]
    if not unexpected.empty:
        raise AssertionError("loan-month ordering produced a non-forward pair")

    observed = adjacent[["_from_state", "_state"]].rename(
        columns={"_state": "to_state"}
    )
    observed = observed.rename(columns={"_from_state": "from_state"})
    long = (
        observed.groupby(["from_state", "to_state"], observed=True, sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    if long.empty:
        long["probability"] = pd.Series(dtype="float64")
    else:
        totals = long.groupby("from_state", observed=True)["count"].transform("sum")
        long["probability"] = long["count"] / totals

    counts = pd.crosstab(observed["from_state"], observed["to_state"])
    counts = counts.reindex(index=order, columns=order, fill_value=0).astype("int64")
    counts.index.name = "from_state"
    counts.columns.name = "to_state"
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    probabilities = counts.div(row_totals, axis=0)
    probabilities.index.name = "from_state"
    probabilities.columns.name = "to_state"

    diagnostics = TransitionDiagnostics(
        input_rows=input_rows,
        input_loans=input_loans,
        unique_loan_months=unique_loan_months,
        usable_loan_months=usable_loan_months,
        duplicate_rows=duplicate_rows,
        duplicate_loan_months=duplicate_loan_months,
        identical_duplicate_loan_months=identical_duplicate_loan_months,
        identical_duplicate_rows_collapsed=identical_duplicate_rows_collapsed,
        conflicting_duplicate_loan_months=conflicting_duplicate_loan_months,
        conflicting_rows_dropped=conflicting_rows_dropped,
        loans_with_duplicates=loans_with_duplicates,
        candidate_pairs=len(candidates),
        adjacent_pairs=len(adjacent),
        gap_pairs_skipped=len(gaps),
        loans_with_gaps=int(gaps[loan_id_col].nunique()),
        largest_gap_months=int(gaps["_month_delta"].max()) if len(gaps) else 0,
        unknown_state_rows=unknown_state_rows,
    )
    return TransitionResult(
        counts=counts,
        probabilities=probabilities,
        long=long,
        diagnostics=diagnostics,
    )


def estimate_monthly_transitions(*args: Any, **kwargs: Any) -> TransitionResult:
    """Alias that makes the required monthly cadence explicit at call sites."""
    return estimate_transitions(*args, **kwargs)

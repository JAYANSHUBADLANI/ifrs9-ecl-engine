"""Canonical monthly loan states for Freddie Mac release 47 records.

The monthly delinquency field describes payment status while the zero balance code
describes why the loan left the active population. A terminal zero balance code takes
priority on the record where it appears. Missing zero balance codes do not alter the
delinquency state.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real
from typing import Any

import pandas as pd

CURRENT = "CURRENT"
DPD_30 = "30_DPD"
DPD_60 = "60_DPD"
DPD_90_PLUS = "90_PLUS"
PAID_OFF = "PAID_OFF"
DEFAULTED = "DEFAULTED"
CENSORED_RPL = "CENSORED_RPL"
CENSORED_DEFECT = "CENSORED_DEFECT"
UNKNOWN = "UNKNOWN"

STATE_ORDER = (
    CURRENT,
    DPD_30,
    DPD_60,
    DPD_90_PLUS,
    PAID_OFF,
    DEFAULTED,
    CENSORED_RPL,
    CENSORED_DEFECT,
    UNKNOWN,
)

TERMINAL_STATES = frozenset(
    {PAID_OFF, DEFAULTED, CENSORED_RPL, CENSORED_DEFECT}
)
CENSORED_STATES = frozenset({CENSORED_RPL, CENSORED_DEFECT})

_ZERO_BALANCE_STATES = {
    "01": PAID_OFF,
    "02": DEFAULTED,
    "03": DEFAULTED,
    "09": DEFAULTED,
    "15": DEFAULTED,
    "16": CENSORED_RPL,
    "96": CENSORED_DEFECT,
    "98": CENSORED_DEFECT,
}

_STATE_ALIASES = {
    "CURRENT": CURRENT,
    "0": CURRENT,
    "00": CURRENT,
    "30": DPD_30,
    "30_DPD": DPD_30,
    "DPD_30": DPD_30,
    "60": DPD_60,
    "60_DPD": DPD_60,
    "DPD_60": DPD_60,
    "90+": DPD_90_PLUS,
    "90_PLUS": DPD_90_PLUS,
    "DPD_90_PLUS": DPD_90_PLUS,
    "PAID_OFF": PAID_OFF,
    "PAIDOFF": PAID_OFF,
    "DEFAULT": DEFAULTED,
    "DEFAULTED": DEFAULTED,
    "REO": DEFAULTED,
    "CENSORED_RPL": CENSORED_RPL,
    "CENSORED_DEFECT": CENSORED_DEFECT,
    "UNKNOWN": UNKNOWN,
    "XX": UNKNOWN,
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, bool) and missing


def _normalise_code(value: Any) -> str | None:
    """Return an upper-case, two-character code where numeric input permits it."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, Real):
        numeric = float(value)
        if numeric.is_integer() and numeric >= 0:
            return f"{int(numeric):02d}"

    text = str(value).strip().upper()
    if text in {"", "<NA>", "NAN", "NONE", "NULL"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(2)
    return text


def map_delinquency_state(code: Any) -> str:
    """Map a release 47 delinquency code to one canonical monthly state.

    Numeric codes 03 and above are 90 days or more past due. ``RA`` is a
    default or REO record. ``XX``, missing values, and unsupported codes are
    retained as ``UNKNOWN`` rather than being treated as current.
    """
    normalised = _normalise_code(code)
    if normalised is None or normalised == "XX":
        return UNKNOWN
    if normalised == "RA":
        return DEFAULTED
    if normalised == "03+":
        return DPD_90_PLUS
    if normalised.isdigit():
        months_past_due = int(normalised)
        if months_past_due == 0:
            return CURRENT
        if months_past_due == 1:
            return DPD_30
        if months_past_due == 2:
            return DPD_60
        if months_past_due >= 3:
            return DPD_90_PLUS
    return UNKNOWN


def map_zero_balance_state(code: Any) -> str | None:
    """Map a release 47 zero balance code to a terminal or censoring state.

    A missing code or explicit ``00`` means the loan has not exited and returns
    ``None``. Unsupported nonmissing codes return ``UNKNOWN`` so data quality
    problems are visible.
    """
    normalised = _normalise_code(code)
    if normalised is None or normalised == "00":
        return None
    return _ZERO_BALANCE_STATES.get(normalised, UNKNOWN)


def coerce_state(value: Any) -> str:
    """Normalise a canonical state label and reject unsupported labels."""
    if _is_missing(value):
        return UNKNOWN
    label = str(value).strip().upper()
    try:
        return _STATE_ALIASES[label]
    except KeyError as exc:
        raise ValueError(f"unsupported loan state: {value!r}") from exc


def map_terminal_state(current_state: Any, zero_balance_code: Any) -> str:
    """Apply a zero balance exit code to an already mapped delinquency state."""
    terminal_state = map_zero_balance_state(zero_balance_code)
    if terminal_state is None:
        return coerce_state(current_state)
    return terminal_state


def derive_state(delinquency_code: Any, zero_balance_code: Any = None) -> str:
    """Derive the canonical state for one monthly performance record."""
    return map_terminal_state(
        map_delinquency_state(delinquency_code), zero_balance_code
    )


def derive_state_series(
    delinquency_codes: Iterable[Any], zero_balance_codes: Iterable[Any]
) -> pd.Series:
    """Vector-style state derivation with positional, not index, alignment."""
    delinquency = list(delinquency_codes)
    zero_balance = list(zero_balance_codes)
    if len(delinquency) != len(zero_balance):
        raise ValueError("delinquency and zero balance inputs must have equal length")

    index = (
        delinquency_codes.index
        if isinstance(delinquency_codes, pd.Series)
        else pd.RangeIndex(len(delinquency))
    )
    values = [
        derive_state(dq_code, zero_code)
        for dq_code, zero_code in zip(delinquency, zero_balance, strict=True)
    ]
    return pd.Series(values, index=index, dtype="string", name="state")


def add_state_column(
    frame: pd.DataFrame,
    delinquency_col: str,
    zero_balance_col: str,
    output_col: str = "state",
) -> pd.DataFrame:
    """Return a copy of a performance frame with its canonical state attached."""
    missing = [
        column
        for column in (delinquency_col, zero_balance_col)
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"missing state input columns: {missing}")
    result = frame.copy()
    result[output_col] = derive_state_series(
        result[delinquency_col], result[zero_balance_col]
    )
    return result


# A descriptive alias for callers working directly with performance records.
derive_performance_state = derive_state


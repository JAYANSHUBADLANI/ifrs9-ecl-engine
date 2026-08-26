"""Scheduled mortgage balance and exposure at default utilities.

The mortgage projection uses contractual fixed-rate amortization. Exposure at
default is then assembled from explicitly named components so non-interest-
bearing principal and unpaid amounts cannot disappear inside a generic balance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EADComponents:
    """Auditable components of exposure at default."""

    interest_bearing_upb: float
    non_interest_bearing_upb: float
    delinquent_accrued_interest: float
    other_unpaid_amounts: float
    undrawn_commitment: float
    credit_conversion_factor: float
    undrawn_ead: float
    total_ead: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative_number(value: Any, *, name: str) -> float:
    parsed = _finite_number(value, name=name)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _whole_months(value: Any, *, name: str, allow_zero: bool = True) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise TypeError(f"{name} must be an integer")
        parsed = int(numeric)
    elif isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except ValueError as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise TypeError(f"{name} must be an integer")
        parsed = int(numeric)
    else:
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        comparator = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparator}")
    return parsed


def _annual_rate(value: Any, *, rate_in_percent: bool) -> float:
    rate = _finite_number(value, name="annual_interest_rate")
    if rate_in_percent:
        rate /= 100.0
    if rate < 0:
        raise ValueError("annual_interest_rate must be nonnegative")
    return rate


def fixed_rate_monthly_payment(
    principal: Any,
    annual_interest_rate: Any,
    remaining_term_months: Any,
    *,
    rate_in_percent: bool = False,
) -> float:
    """Return the level monthly payment for a fully amortizing fixed-rate loan.

    The annual rate is a decimal by default, so 6 percent is supplied as 0.06.
    Set ``rate_in_percent=True`` to supply 6 instead. A zero-rate mortgage pays
    equal principal each month.
    """

    balance = _nonnegative_number(principal, name="principal")
    term = _whole_months(
        remaining_term_months, name="remaining_term_months", allow_zero=True
    )
    annual_rate = _annual_rate(annual_interest_rate, rate_in_percent=rate_in_percent)
    if balance == 0:
        return 0.0
    if term == 0:
        raise ValueError("remaining_term_months must be positive when principal is positive")
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        return balance / term
    denominator = -np.expm1(-term * np.log1p(monthly_rate))
    return float(balance * monthly_rate / denominator)


def scheduled_balance_at_month(
    principal: Any,
    annual_interest_rate: Any,
    remaining_term_months: Any,
    month: Any,
    *,
    rate_in_percent: bool = False,
) -> float:
    """Return scheduled principal after ``month`` level payments."""

    balance = _nonnegative_number(principal, name="principal")
    term = _whole_months(
        remaining_term_months, name="remaining_term_months", allow_zero=True
    )
    elapsed = _whole_months(month, name="month", allow_zero=True)
    annual_rate = _annual_rate(annual_interest_rate, rate_in_percent=rate_in_percent)
    if balance == 0:
        return 0.0
    if term == 0:
        raise ValueError("remaining_term_months must be positive when principal is positive")
    if elapsed == 0:
        return balance
    if elapsed >= term:
        return 0.0
    payment = fixed_rate_monthly_payment(
        balance,
        annual_rate,
        term,
        rate_in_percent=False,
    )
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        projected = balance - payment * elapsed
    else:
        growth_minus_one = np.expm1(elapsed * np.log1p(monthly_rate))
        projected = balance * (growth_minus_one + 1.0) - (
            payment * growth_minus_one / monthly_rate
        )
    rounding_tolerance = max(1.0, balance) * 1e-12
    if abs(projected) <= rounding_tolerance:
        return 0.0
    return float(max(projected, 0.0))


def project_scheduled_balances(
    principal: Any,
    annual_interest_rate: Any,
    remaining_term_months: Any,
    *,
    horizon_months: Any | None = None,
    rate_in_percent: bool = False,
    include_initial: bool = True,
) -> np.ndarray:
    """Project scheduled principal balances from today through a horizon.

    By default the projection runs to contractual maturity. If the requested
    horizon extends beyond maturity, the returned balances remain zero. With
    ``include_initial=True``, element zero is today's principal.
    """

    term = _whole_months(
        remaining_term_months, name="remaining_term_months", allow_zero=True
    )
    horizon = (
        term
        if horizon_months is None
        else _whole_months(horizon_months, name="horizon_months", allow_zero=True)
    )
    start = 0 if include_initial else 1
    if horizon < start:
        return np.array([], dtype="float64")
    return np.asarray(
        [
            scheduled_balance_at_month(
                principal,
                annual_interest_rate,
                term,
                month,
                rate_in_percent=rate_in_percent,
            )
            for month in range(start, horizon + 1)
        ],
        dtype="float64",
    )


def scheduled_amortization_schedule(
    principal: Any,
    annual_interest_rate: Any,
    remaining_term_months: Any,
    *,
    horizon_months: Any | None = None,
    rate_in_percent: bool = False,
) -> pd.DataFrame:
    """Return a month-by-month scheduled fixed-rate amortization table."""

    balance = _nonnegative_number(principal, name="principal")
    term = _whole_months(
        remaining_term_months, name="remaining_term_months", allow_zero=True
    )
    horizon = (
        term
        if horizon_months is None
        else _whole_months(horizon_months, name="horizon_months", allow_zero=True)
    )
    annual_rate = _annual_rate(annual_interest_rate, rate_in_percent=rate_in_percent)
    payment = fixed_rate_monthly_payment(
        balance,
        annual_rate,
        term,
        rate_in_percent=False,
    )
    monthly_rate = annual_rate / 12.0
    rows: list[dict[str, float | int]] = []
    opening = balance
    for payment_month in range(1, horizon + 1):
        if payment_month <= term and opening > 0:
            interest = opening * monthly_rate
            cash_payment = min(payment, opening + interest)
            principal_payment = cash_payment - interest
            closing = max(opening - principal_payment, 0.0)
            if payment_month == term:
                principal_payment = opening
                cash_payment = opening + interest
                closing = 0.0
        else:
            interest = 0.0
            cash_payment = 0.0
            principal_payment = 0.0
            closing = 0.0
        rows.append(
            {
                "payment_month": payment_month,
                "opening_balance": float(opening),
                "scheduled_payment": float(cash_payment),
                "interest": float(interest),
                "principal": float(principal_payment),
                "closing_balance": float(closing),
            }
        )
        opening = closing
    return pd.DataFrame(
        rows,
        columns=[
            "payment_month",
            "opening_balance",
            "scheduled_payment",
            "interest",
            "principal",
            "closing_balance",
        ],
    )


def default_ead_components(
    interest_bearing_upb: Any,
    *,
    non_interest_bearing_upb: Any = 0.0,
    delinquent_accrued_interest: Any = 0.0,
    other_unpaid_amounts: Any = 0.0,
    undrawn_commitment: Any = 0.0,
    credit_conversion_factor: Any = 0.0,
) -> EADComponents:
    """Assemble EAD without omitting deferred principal or unpaid amounts.

    ``interest_bearing_upb`` and ``non_interest_bearing_upb`` must be separate
    components. If a source field already represents their total, pass that
    total as ``interest_bearing_upb`` and leave ``non_interest_bearing_upb`` at
    zero to avoid double counting.
    """

    interest_bearing = _nonnegative_number(
        interest_bearing_upb, name="interest_bearing_upb"
    )
    non_interest_bearing = _nonnegative_number(
        non_interest_bearing_upb, name="non_interest_bearing_upb"
    )
    accrued_interest = _nonnegative_number(
        delinquent_accrued_interest, name="delinquent_accrued_interest"
    )
    unpaid = _nonnegative_number(other_unpaid_amounts, name="other_unpaid_amounts")
    undrawn = _nonnegative_number(undrawn_commitment, name="undrawn_commitment")
    ccf = _finite_number(credit_conversion_factor, name="credit_conversion_factor")
    if not 0 <= ccf <= 1:
        raise ValueError("credit_conversion_factor must be between zero and one")
    undrawn_ead = undrawn * ccf
    total = interest_bearing + non_interest_bearing + accrued_interest + unpaid + undrawn_ead
    return EADComponents(
        interest_bearing_upb=interest_bearing,
        non_interest_bearing_upb=non_interest_bearing,
        delinquent_accrued_interest=accrued_interest,
        other_unpaid_amounts=unpaid,
        undrawn_commitment=undrawn,
        credit_conversion_factor=ccf,
        undrawn_ead=undrawn_ead,
        total_ead=total,
    )


def calculate_default_ead(
    interest_bearing_upb: Any,
    *,
    non_interest_bearing_upb: Any = 0.0,
    delinquent_accrued_interest: Any = 0.0,
    other_unpaid_amounts: Any = 0.0,
    undrawn_commitment: Any = 0.0,
    credit_conversion_factor: Any = 0.0,
) -> float:
    """Return total EAD while retaining a component API for audit use."""

    return default_ead_components(
        interest_bearing_upb,
        non_interest_bearing_upb=non_interest_bearing_upb,
        delinquent_accrued_interest=delinquent_accrued_interest,
        other_unpaid_amounts=other_unpaid_amounts,
        undrawn_commitment=undrawn_commitment,
        credit_conversion_factor=credit_conversion_factor,
    ).total_ead


def _component_path(value: Any, *, length: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        parsed = _nonnegative_number(value, name=name)
        return np.full(length, parsed, dtype="float64")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be numeric or a sequence") from exc
    if len(values) != length:
        raise ValueError(f"{name} path must contain {length} values")
    return np.asarray(
        [_nonnegative_number(item, name=f"{name}[{index}]") for index, item in enumerate(values)],
        dtype="float64",
    )


def project_amortizing_ead(
    principal: Any,
    annual_interest_rate: Any,
    remaining_term_months: Any,
    *,
    horizon_months: Any | None = None,
    rate_in_percent: bool = False,
    non_interest_bearing_upb: Any = 0.0,
    delinquent_accrued_interest: Any = 0.0,
    other_unpaid_amounts: Any = 0.0,
    undrawn_commitment: Any = 0.0,
    credit_conversion_factor: Any = 0.0,
) -> pd.DataFrame:
    """Project monthly EAD using scheduled mortgage principal and add-ons.

    Each add-on can be a scalar, which is held constant, or a sequence with one
    value for every projection month including month zero. This makes assumptions
    about deferred principal and accrued amounts visible to callers.
    """

    term = _whole_months(
        remaining_term_months, name="remaining_term_months", allow_zero=True
    )
    horizon = (
        term
        if horizon_months is None
        else _whole_months(horizon_months, name="horizon_months", allow_zero=True)
    )
    balances = project_scheduled_balances(
        principal,
        annual_interest_rate,
        term,
        horizon_months=horizon,
        rate_in_percent=rate_in_percent,
        include_initial=True,
    )
    length = horizon + 1
    non_interest = _component_path(
        non_interest_bearing_upb, length=length, name="non_interest_bearing_upb"
    )
    accrued = _component_path(
        delinquent_accrued_interest,
        length=length,
        name="delinquent_accrued_interest",
    )
    unpaid = _component_path(
        other_unpaid_amounts, length=length, name="other_unpaid_amounts"
    )
    undrawn = _component_path(
        undrawn_commitment, length=length, name="undrawn_commitment"
    )
    ccf = _finite_number(credit_conversion_factor, name="credit_conversion_factor")
    if not 0 <= ccf <= 1:
        raise ValueError("credit_conversion_factor must be between zero and one")
    undrawn_ead = undrawn * ccf
    total = balances + non_interest + accrued + unpaid + undrawn_ead
    return pd.DataFrame(
        {
            "month": np.arange(length, dtype="int64"),
            "interest_bearing_upb": balances,
            "non_interest_bearing_upb": non_interest,
            "delinquent_accrued_interest": accrued,
            "other_unpaid_amounts": unpaid,
            "undrawn_ead": undrawn_ead,
            "default_ead": total,
        }
    )


def add_default_ead(
    data: pd.DataFrame,
    *,
    interest_bearing_col: str = "current_interest_bearing_upb",
    non_interest_bearing_col: str = "current_non_interest_bearing_upb",
    accrued_interest_col: str = "delinquent_accrued_interest",
    other_unpaid_col: str | None = None,
    output_col: str = "default_ead",
) -> pd.DataFrame:
    """Return a copy with row-level EAD from Freddie Mac performance fields."""

    required = [interest_bearing_col, non_interest_bearing_col, accrued_interest_col]
    if other_unpaid_col is not None:
        required.append(other_unpaid_col)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"missing EAD input columns: {missing}")
    result = data.copy()
    totals: list[float] = []
    for _, row in result.iterrows():
        unpaid = 0.0 if other_unpaid_col is None else row[other_unpaid_col]
        totals.append(
            calculate_default_ead(
                row[interest_bearing_col],
                non_interest_bearing_upb=row[non_interest_bearing_col],
                delinquent_accrued_interest=row[accrued_interest_col],
                other_unpaid_amounts=unpaid,
            )
        )
    result[output_col] = totals
    return result


def default_ead_from_record(
    record: Mapping[str, Any],
    *,
    interest_bearing_field: str = "current_interest_bearing_upb",
    non_interest_bearing_field: str = "current_non_interest_bearing_upb",
    accrued_interest_field: str = "delinquent_accrued_interest",
    other_unpaid_field: str | None = None,
) -> EADComponents:
    """Assemble an auditable EAD breakdown from one performance record."""

    missing = [
        field
        for field in (
            interest_bearing_field,
            non_interest_bearing_field,
            accrued_interest_field,
        )
        if field not in record
    ]
    if other_unpaid_field is not None and other_unpaid_field not in record:
        missing.append(other_unpaid_field)
    if missing:
        raise KeyError(f"missing EAD record fields: {missing}")
    return default_ead_components(
        record[interest_bearing_field],
        non_interest_bearing_upb=record[non_interest_bearing_field],
        delinquent_accrued_interest=record[accrued_interest_field],
        other_unpaid_amounts=(
            0.0 if other_unpaid_field is None else record[other_unpaid_field]
        ),
    )


fixed_rate_payment = fixed_rate_monthly_payment
scheduled_fixed_rate_balance = scheduled_balance_at_month
project_scheduled_balance = project_scheduled_balances
project_fixed_rate_mortgage_balance = project_scheduled_balances
scheduled_balance_projection = project_scheduled_balances
build_amortization_schedule = scheduled_amortization_schedule
default_ead = calculate_default_ead
compute_default_ead = calculate_default_ead
amortizing_ead = project_amortizing_ead

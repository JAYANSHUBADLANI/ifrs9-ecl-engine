"""Monthly discounted expected credit loss calculations.

Stage 1 uses marginal defaults in months 1 to 12. Stage 2 uses all supplied
months. Stage 3 either assumes immediate default with a 100 percent marginal
PD or accepts a direct expected cash-shortfall proxy. Annual rates are treated
as effective annual rates and converted consistently through fractional years.
LGD must be between zero and one, and EAD must be nonnegative.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .staging import Stage, normalize_stage


@dataclass(frozen=True)
class ECLResult:
    """Auditable monthly ECL schedule and its discounted total."""

    stage: Stage
    method: str
    months: np.ndarray
    marginal_pd: np.ndarray
    lgd: np.ndarray
    ead: np.ndarray
    discount_factors: np.ndarray
    undiscounted_monthly_ecl: np.ndarray
    discounted_monthly_ecl: np.ndarray
    total_undiscounted_ecl: float
    total_ecl: float

    @property
    def horizon_months(self) -> int:
        """Latest modeled loss month, with zero denoting immediate Stage 3 loss."""
        return int(self.months[-1]) if len(self.months) else 0

    def to_frame(self) -> pd.DataFrame:
        """Return the calculation schedule as a new data frame."""
        return pd.DataFrame(
            {
                "month": self.months.copy(),
                "marginal_pd": self.marginal_pd.copy(),
                "lgd": self.lgd.copy(),
                "ead": self.ead.copy(),
                "discount_factor": self.discount_factors.copy(),
                "undiscounted_ecl": self.undiscounted_monthly_ecl.copy(),
                "discounted_ecl": self.discounted_monthly_ecl.copy(),
            }
        )


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _as_one_dimensional(values: Any, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be numeric")
    if isinstance(values, Real) and not isinstance(values, (bool, np.bool_)):
        array = np.asarray([values], dtype="float64")
    else:
        try:
            if isinstance(values, np.ndarray):
                array = np.asarray(values, dtype="float64")
            else:
                array = np.asarray(list(values), dtype="float64")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a numeric scalar or sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _select_term_array(
    values: Any,
    full_length: int,
    horizon: int,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> np.ndarray:
    """Broadcast a scalar or select the horizon from a supplied term curve."""
    array = _as_one_dimensional(values, name)
    if len(array) == 1:
        array = np.repeat(array, horizon)
    elif len(array) == full_length:
        array = array[:horizon]
    elif len(array) != horizon:
        raise ValueError(f"{name} must be scalar or have {horizon} or {full_length} values")
    if (array < minimum).any():
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and (array > maximum).any():
        raise ValueError(f"{name} must be at most {maximum}")
    return array.astype("float64", copy=False)


def _first_term_value(
    values: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> np.ndarray:
    array = _as_one_dimensional(values, name)
    if not len(array):
        raise ValueError(f"{name} must contain at least one value")
    value = array[:1]
    if value[0] < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value[0] > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value.astype("float64", copy=False)


def validate_marginal_pd(marginal_pd: Any) -> np.ndarray:
    """Validate and return a copy of a marginal default probability curve."""
    curve = _as_one_dimensional(marginal_pd, "marginal_pd")
    if not len(curve):
        raise ValueError("marginal_pd must contain at least one month")
    tolerance = 1e-12
    if (curve < -tolerance).any() or (curve > 1.0 + tolerance).any():
        raise ValueError("marginal_pd values must be between 0 and 1")
    if curve.sum() > 1.0 + tolerance:
        raise ValueError("marginal_pd values must sum to no more than 1")
    return np.clip(curve, 0.0, 1.0).astype("float64", copy=True)


def discount_factors(months: Any, effective_annual_rate: float) -> np.ndarray:
    """Calculate present-value factors for integer months after reporting date."""
    rate = _finite_scalar(effective_annual_rate, "effective_annual_rate")
    if rate <= -1.0:
        raise ValueError("effective_annual_rate must be greater than -1")
    month_array = _as_one_dimensional(months, "months")
    if (month_array < 0).any() or not np.equal(month_array, np.floor(month_array)).all():
        raise ValueError("months must contain nonnegative integers")
    return np.power(1.0 + rate, -month_array / 12.0)


def _stage3_result(
    lgd: Any,
    ead: Any,
    effective_annual_rate: float,
    cash_shortfall: float | None,
    cash_shortfall_month: int,
) -> ECLResult:
    if isinstance(cash_shortfall_month, (bool, np.bool_)):
        raise TypeError("stage3_cash_shortfall_month must be an integer")
    month_value = _finite_scalar(cash_shortfall_month, "stage3_cash_shortfall_month")
    if month_value < 0 or not month_value.is_integer():
        raise ValueError("stage3_cash_shortfall_month must be a nonnegative integer")
    month = int(month_value)

    if cash_shortfall is not None:
        shortfall = _finite_scalar(cash_shortfall, "stage3_cash_shortfall")
        if shortfall < 0:
            raise ValueError("stage3_cash_shortfall must be nonnegative")
        selected_lgd = np.asarray([1.0])
        selected_ead = np.asarray([shortfall])
        method = "cash_shortfall_proxy"
    else:
        if lgd is None or ead is None:
            raise ValueError("Stage 3 requires lgd and ead or stage3_cash_shortfall")
        selected_lgd = _first_term_value(lgd, "lgd", minimum=0.0, maximum=1.0)
        selected_ead = _first_term_value(ead, "ead", minimum=0.0)
        method = "immediate_default"
        month = 0

    months = np.asarray([month], dtype="int64")
    marginal_pd = np.asarray([1.0])
    factors = discount_factors(months, effective_annual_rate)
    undiscounted = marginal_pd * selected_lgd * selected_ead
    discounted = undiscounted * factors
    return ECLResult(
        stage=Stage.STAGE_3,
        method=method,
        months=months,
        marginal_pd=marginal_pd,
        lgd=selected_lgd,
        ead=selected_ead,
        discount_factors=factors,
        undiscounted_monthly_ecl=undiscounted,
        discounted_monthly_ecl=discounted,
        total_undiscounted_ecl=float(undiscounted.sum()),
        total_ecl=float(discounted.sum()),
    )


def calculate_ecl(
    marginal_pd: Any,
    lgd: Any,
    ead: Any,
    *,
    stage: Any,
    effective_annual_rate: float = 0.0,
    stage3_cash_shortfall: float | None = None,
    stage3_cash_shortfall_month: int = 0,
) -> ECLResult:
    """Calculate 12-month, lifetime, or credit-impaired ECL.

    Marginal PD is the unconditional probability of first default in each
    month. LGD and EAD can be scalars or monthly sequences. A scalar is held
    constant over the selected horizon. Stage 3 ignores the supplied PD curve.
    Its default method applies a 100 percent PD immediately to the first LGD
    and EAD values. Alternatively, ``stage3_cash_shortfall`` supplies the loss
    amount directly, with an optional expected realization month.
    """
    selected_stage = normalize_stage(stage)
    rate = _finite_scalar(effective_annual_rate, "effective_annual_rate")
    if rate <= -1.0:
        raise ValueError("effective_annual_rate must be greater than -1")

    if selected_stage is Stage.STAGE_3:
        return _stage3_result(
            lgd=lgd,
            ead=ead,
            effective_annual_rate=rate,
            cash_shortfall=stage3_cash_shortfall,
            cash_shortfall_month=stage3_cash_shortfall_month,
        )

    full_curve = validate_marginal_pd(marginal_pd)
    full_length = len(full_curve)
    horizon = min(12, full_length) if selected_stage is Stage.STAGE_1 else full_length
    curve = full_curve[:horizon]
    selected_lgd = _select_term_array(
        lgd,
        full_length,
        horizon,
        "lgd",
        minimum=0.0,
        maximum=1.0,
    )
    selected_ead = _select_term_array(ead, full_length, horizon, "ead", minimum=0.0)
    months = np.arange(1, horizon + 1, dtype="int64")
    factors = discount_factors(months, rate)
    undiscounted = curve * selected_lgd * selected_ead
    discounted = undiscounted * factors

    return ECLResult(
        stage=selected_stage,
        method="12_month_ecl" if selected_stage is Stage.STAGE_1 else "lifetime_ecl",
        months=months,
        marginal_pd=curve,
        lgd=selected_lgd,
        ead=selected_ead,
        discount_factors=factors,
        undiscounted_monthly_ecl=undiscounted,
        discounted_monthly_ecl=discounted,
        total_undiscounted_ecl=float(undiscounted.sum()),
        total_ecl=float(discounted.sum()),
    )


def _optional_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return value


def calculate_portfolio_ecl(
    exposures: pd.DataFrame,
    *,
    marginal_pd_col: str = "marginal_pd",
    lgd_col: str = "lgd",
    ead_col: str = "ead",
    stage_col: str = "stage",
    effective_rate_col: str | None = "effective_annual_rate",
    effective_annual_rate: float = 0.0,
    stage3_cash_shortfall_col: str | None = None,
    ecl_col: str = "ecl",
) -> pd.DataFrame:
    """Calculate ECL row by row and return a copy of the exposure frame.

    Curve-valued cells may contain lists, tuples, NumPy arrays, or pandas
    Series. If ``effective_rate_col`` is absent or ``None``, the scalar rate is
    used for every exposure.
    """
    required = [marginal_pd_col, lgd_col, ead_col, stage_col]
    if effective_rate_col is not None and effective_rate_col in exposures.columns:
        required.append(effective_rate_col)
    if stage3_cash_shortfall_col is not None:
        required.append(stage3_cash_shortfall_col)
    missing = [column for column in required if column not in exposures.columns]
    if missing:
        raise KeyError(f"missing ECL input columns: {missing}")

    result = exposures.copy(deep=True)
    totals: list[float] = []
    horizons: list[int] = []
    methods: list[str] = []
    for _, row in result.iterrows():
        row_rate = (
            row[effective_rate_col]
            if effective_rate_col is not None and effective_rate_col in result.columns
            else effective_annual_rate
        )
        cash_shortfall = (
            _optional_scalar(row[stage3_cash_shortfall_col])
            if stage3_cash_shortfall_col is not None
            else None
        )
        calculation = calculate_ecl(
            row[marginal_pd_col],
            row[lgd_col],
            row[ead_col],
            stage=row[stage_col],
            effective_annual_rate=row_rate,
            stage3_cash_shortfall=cash_shortfall,
        )
        totals.append(calculation.total_ecl)
        horizons.append(calculation.horizon_months)
        methods.append(calculation.method)

    result[ecl_col] = pd.Series(totals, index=result.index, dtype="float64")
    result[f"{ecl_col}_horizon_months"] = pd.Series(horizons, index=result.index, dtype="int64")
    result[f"{ecl_col}_method"] = pd.Series(methods, index=result.index, dtype="object")
    return result

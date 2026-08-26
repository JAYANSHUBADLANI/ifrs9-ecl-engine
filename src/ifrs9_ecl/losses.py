"""Release 47 actual loss reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping


ACTUAL_LOSS_COMPONENT_FIELDS = (
    "zero_balance_removal_upb",
    "net_sales_proceeds",
    "delinquent_accrued_interest",
    "total_expenses",
    "mi_recoveries",
    "non_mi_recoveries",
)


@dataclass(frozen=True)
class ActualLossReconciliation:
    """Diagnostics from reconciling disclosed and recomputed actual loss."""

    actual_loss: Decimal | None
    recomputed_loss: Decimal | None
    difference: Decimal | None
    absolute_difference: Decimal | None
    tolerance: Decimal
    is_within_tolerance: bool | None
    status: str
    components: Mapping[str, Decimal]


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return value != value
    return False


def _as_decimal(value: object, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, received {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite, received {value!r}")
    return parsed


def reconcile_actual_loss(
    record: Mapping[str, object],
    *,
    tolerance: object = Decimal("0.01"),
) -> ActualLossReconciliation:
    """Recompute and reconcile Release 47 actual loss for one record.

    Release 47 stores recoveries and gains as negative amounts and expenses
    and losses as positive amounts. The six inputs are therefore added. Blank
    components are treated as zero only when disclosed actual loss is present.
    Expense breakdown fields are intentionally ignored because total_expenses
    already contains their sum.

    The difference is recomputed loss minus disclosed actual loss. A missing
    disclosed actual loss returns a skipped result because Freddie Mac leaves
    that field null for records where the calculation is not applicable or is
    not yet available.
    """
    parsed_tolerance = _as_decimal(tolerance, field="tolerance")
    if parsed_tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    disclosed = record.get("actual_loss")
    if _is_blank(disclosed):
        return ActualLossReconciliation(
            actual_loss=None,
            recomputed_loss=None,
            difference=None,
            absolute_difference=None,
            tolerance=parsed_tolerance,
            is_within_tolerance=None,
            status="actual_loss_missing",
            components={},
        )

    actual_loss = _as_decimal(disclosed, field="actual_loss")
    components: dict[str, Decimal] = {}
    for field in ACTUAL_LOSS_COMPONENT_FIELDS:
        value = record.get(field)
        components[field] = (
            Decimal("0") if _is_blank(value) else _as_decimal(value, field=field)
        )

    recomputed_loss = sum(components.values(), start=Decimal("0"))
    difference = recomputed_loss - actual_loss
    absolute_difference = abs(difference)
    is_within_tolerance = absolute_difference <= parsed_tolerance
    status = "reconciled" if is_within_tolerance else "outside_tolerance"

    return ActualLossReconciliation(
        actual_loss=actual_loss,
        recomputed_loss=recomputed_loss,
        difference=difference,
        absolute_difference=absolute_difference,
        tolerance=parsed_tolerance,
        is_within_tolerance=is_within_tolerance,
        status=status,
        components=components,
    )

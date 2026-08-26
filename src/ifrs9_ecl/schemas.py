"""Release 47 raw file schemas for the Freddie Mac SFLLD dataset.

The source files have no header row. These ordered names are therefore part of
the ingestion contract and must stay aligned with the official July 2026 file
layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


RELEASE_NUMBER = 47


@dataclass(frozen=True, slots=True)
class RawFileSchema:
    """An ordered, headerless source-file schema."""

    name: str
    member_prefix: str
    columns: tuple[str, ...]
    loan_id_column: str = "loan_id"
    _column_indexes: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("A raw schema must contain at least one column")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(f"Schema {self.name!r} contains duplicate column names")
        if self.loan_id_column not in self.columns:
            raise ValueError(
                f"Schema {self.name!r} does not contain loan ID column "
                f"{self.loan_id_column!r}"
            )
        object.__setattr__(
            self,
            "_column_indexes",
            MappingProxyType({column: index for index, column in enumerate(self.columns)}),
        )

    @property
    def width(self) -> int:
        """Return the exact number of fields expected in each physical row."""

        return len(self.columns)

    def index(self, column: str) -> int:
        """Return the zero-based position of a standardized column name."""

        try:
            return self._column_indexes[column]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.name} column: {column!r}") from exc


ORIGINATION_COLUMNS: tuple[str, ...] = (
    "classic_fico",
    "first_payment_date",
    "first_time_homebuyer_indicator",
    "maturity_date",
    "msa_or_metropolitan_division",
    "mi_percentage",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_indicator",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_id",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "super_conforming_flag",
    "pre_harp_loan_id",
    "special_eligibility_program",
    "harp_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "vantage_score_4_0",
)


PERFORMANCE_COLUMNS: tuple[str, ...] = (
    "loan_id",
    "reporting_month",
    "current_actual_upb",
    "delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "cumulative_modification_costs",
    "interest_rate_step_indicator",
    "payment_deferral_flag",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_plan",
    "current_period_modification_costs",
    "current_interest_bearing_upb",
    "mi_cancellation_indicator",
    "servicer_name",
    "bankruptcy_cramdown_costs",
)


ORIGINATION_SCHEMA = RawFileSchema(
    name="origination",
    member_prefix="orig_",
    columns=ORIGINATION_COLUMNS,
)

PERFORMANCE_SCHEMA = RawFileSchema(
    name="performance",
    member_prefix="perf_",
    columns=PERFORMANCE_COLUMNS,
)


SCHEMAS_BY_NAME: Mapping[str, RawFileSchema] = MappingProxyType(
    {
        ORIGINATION_SCHEMA.name: ORIGINATION_SCHEMA,
        PERFORMANCE_SCHEMA.name: PERFORMANCE_SCHEMA,
    }
)


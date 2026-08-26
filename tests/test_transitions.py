import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.states import (
    CURRENT,
    DEFAULTED,
    DPD_30,
    DPD_60,
    PAID_OFF,
)
from ifrs9_ecl.transitions import estimate_monthly_transitions, estimate_transitions


def test_counts_and_row_probabilities_use_adjacent_calendar_months():
    panel = pd.DataFrame(
        {
            "loan_id": ["a", "a", "a", "b", "b"],
            "reporting_month": [202001, 202002, 202003, 202001, 202002],
            "state": [CURRENT, DPD_30, CURRENT, CURRENT, DPD_30],
        }
    )
    result = estimate_transitions(panel)

    assert result.counts.loc[CURRENT, DPD_30] == 2
    assert result.counts.loc[DPD_30, CURRENT] == 1
    assert result.probabilities.loc[CURRENT, DPD_30] == 1.0
    assert result.probabilities.loc[DPD_30, CURRENT] == 1.0
    assert result.diagnostics.adjacent_pairs == 3
    assert result.diagnostics.gap_pairs_skipped == 0


def test_a_gap_is_reported_and_never_counted_as_a_transition():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1],
            "reporting_month": ["2020-01", "2020-03", "2020-04"],
            "state": [CURRENT, DPD_60, DEFAULTED],
        }
    )
    result = estimate_transitions(panel)

    assert result.counts.loc[CURRENT].sum() == 0
    assert result.counts.loc[DPD_60, DEFAULTED] == 1
    assert result.diagnostics.candidate_pairs == 2
    assert result.diagnostics.adjacent_pairs == 1
    assert result.diagnostics.gap_pairs_skipped == 1
    assert result.diagnostics.loans_with_gaps == 1
    assert result.diagnostics.largest_gap_months == 2


def test_identical_duplicates_are_collapsed_without_double_counting():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1],
            "reporting_month": [202001, 202001, 202002],
            "state": [CURRENT, CURRENT, DPD_30],
        }
    )
    result = estimate_transitions(panel)

    assert result.counts.loc[CURRENT, DPD_30] == 1
    assert result.diagnostics.duplicate_rows == 1
    assert result.diagnostics.duplicate_loan_months == 1
    assert result.diagnostics.identical_duplicate_loan_months == 1
    assert result.diagnostics.identical_duplicate_rows_collapsed == 1
    assert result.diagnostics.conflicting_duplicate_loan_months == 0


def test_conflicting_duplicate_month_is_excluded_and_creates_a_gap():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1, 1],
            "reporting_month": [202001, 202002, 202002, 202003],
            "state": [CURRENT, DPD_30, DPD_60, DEFAULTED],
        }
    )
    result = estimate_transitions(panel)

    assert result.counts.to_numpy().sum() == 0
    assert result.diagnostics.conflicting_duplicate_loan_months == 1
    assert result.diagnostics.conflicting_rows_dropped == 2
    assert result.diagnostics.usable_loan_months == 2
    assert result.diagnostics.gap_pairs_skipped == 1


def test_terminal_states_are_not_given_synthetic_absorbing_transitions():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 2, 2],
            "reporting_month": [202001, 202002, 202001, 202002],
            "state": [CURRENT, DEFAULTED, CURRENT, PAID_OFF],
        }
    )
    result = estimate_transitions(panel)

    assert result.counts.loc[CURRENT, DEFAULTED] == 1
    assert result.counts.loc[CURRENT, PAID_OFF] == 1
    assert result.counts.loc[DEFAULTED].sum() == 0
    assert result.counts.loc[PAID_OFF].sum() == 0
    assert result.probabilities.loc[DEFAULTED].isna().all()
    assert result.probabilities.loc[PAID_OFF].isna().all()


def test_long_form_probabilities_reconcile_to_matrix():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 2, 2],
            "reporting_month": [202001, 202002, 202001, 202002],
            "state": [CURRENT, CURRENT, CURRENT, DPD_30],
        }
    )
    result = estimate_monthly_transitions(panel)
    current = result.long[result.long["from_state"] == CURRENT].set_index("to_state")

    assert current.loc[CURRENT, "count"] == 1
    assert current.loc[DPD_30, "count"] == 1
    assert current.loc[CURRENT, "probability"] == 0.5
    assert current.loc[DPD_30, "probability"] == 0.5


def test_datetime_months_and_state_aliases_are_accepted():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1],
            "reporting_month": pd.to_datetime(["2020-01-31", "2020-02-01"]),
            "state": ["current", "30"],
        }
    )
    result = estimate_transitions(panel)
    assert result.counts.loc[CURRENT, DPD_30] == 1


def test_invalid_month_and_missing_identifier_fail_loudly():
    bad_month = pd.DataFrame(
        {"loan_id": [1], "reporting_month": [202013], "state": [CURRENT]}
    )
    with pytest.raises(ValueError, match="reporting month"):
        estimate_transitions(bad_month)

    missing_id = pd.DataFrame(
        {"loan_id": [np.nan], "reporting_month": [202001], "state": [CURRENT]}
    )
    with pytest.raises(ValueError, match="identifiers"):
        estimate_transitions(missing_id)


def test_input_frame_is_not_mutated():
    panel = pd.DataFrame(
        {"loan_id": [1], "reporting_month": [202001], "state": [CURRENT]}
    )
    before = panel.copy(deep=True)
    estimate_transitions(panel)
    pd.testing.assert_frame_equal(panel, before)


def test_empty_panel_returns_empty_matrices_and_zero_diagnostics():
    panel = pd.DataFrame(columns=["loan_id", "reporting_month", "state"])
    result = estimate_transitions(panel)

    assert result.counts.to_numpy().sum() == 0
    assert result.probabilities.isna().all().all()
    assert result.long.empty
    assert result.diagnostics.input_rows == 0
    assert result.diagnostics.adjacent_pairs == 0


def test_custom_state_order_cannot_silently_drop_an_observed_state():
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1],
            "reporting_month": [202001, 202002],
            "state": [CURRENT, DPD_30],
        }
    )
    with pytest.raises(ValueError, match="omits observed states"):
        estimate_transitions(panel, state_order=[CURRENT])

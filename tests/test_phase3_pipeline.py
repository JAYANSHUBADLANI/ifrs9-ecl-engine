"""Tests for the executable Phase 3 integration pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.lgd import (
    CureDefinition,
    fit_conditional_severity_model,
    fit_weighted_cure_model,
)
from ifrs9_ecl.phase3 import (
    apply_cure_adjusted_lgd,
    add_months,
    build_ead_validation_rows,
    evaluate_cure_predictions,
    evaluate_ead_validation,
    evaluate_severity_predictions,
    extract_realized_lgd_rows,
    month_ordinal,
    phase3_run_signature,
    prepare_cure_datasets,
    prepare_snapshot_inputs,
    project_snapshot_ead_sample,
    stable_row_sample,
    validate_release47_terminal_mapping,
    Phase3Settings,
)
from ifrs9_ecl.states import CENSORED_RPL, CURRENT, DEFAULTED, DPD_30, DPD_60


def test_month_helpers_cross_year_and_reject_invalid_months() -> None:
    assert add_months(202012, 1) == 202101
    assert add_months(202101, -2) == 202011
    assert month_ordinal(202102) - month_ordinal(202012) == 2
    with pytest.raises(ValueError, match="valid month"):
        month_ordinal(202113)
    with pytest.raises(TypeError, match="integer"):
        add_months(202101, 1.5)  # type: ignore[arg-type]


def test_release47_code15_is_default_and_code16_is_censor() -> None:
    panel = pd.DataFrame(
        {
            "zero_balance_code": ["15", "16", "", "02"],
            "state": [DEFAULTED, CENSORED_RPL, CURRENT, DEFAULTED],
        }
    )

    diagnostics = validate_release47_terminal_mapping(panel)

    assert diagnostics == {
        "code15_default_rows": 1,
        "code15_misclassified_rows": 0,
        "code16_censor_rows": 1,
        "code16_misclassified_rows": 0,
    }
    bad = panel.copy()
    bad.loc[0, "state"] = CENSORED_RPL
    with pytest.raises(ValueError, match="code 15"):
        validate_release47_terminal_mapping(bad)


def test_cure_dataset_uses_three_adjacent_current_months_and_split_cutoffs() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [
                "train_cure",
                "train_cure",
                "train_cure",
                "train_cure",
                "train_default",
                "train_default",
                "validation_cure",
                "validation_cure",
                "validation_cure",
                "validation_cure",
                "validation_cure",
                "validation_rpl",
                "validation_rpl",
                "validation_rpl",
            ],
            "reporting_month": [
                202001,
                202002,
                202003,
                202004,
                202001,
                202002,
                202012,
                202101,
                202102,
                202103,
                202104,
                202012,
                202101,
                202102,
            ],
            "state": [
                DPD_30,
                CURRENT,
                CURRENT,
                CURRENT,
                DPD_60,
                DEFAULTED,
                CURRENT,
                DPD_30,
                CURRENT,
                CURRENT,
                CURRENT,
                CURRENT,
                DPD_30,
                CENSORED_RPL,
            ],
            "vintage": ["2020Q1"] * 14,
        }
    )

    episodes, diagnostics = prepare_cure_datasets(
        panel,
        development_end_month=202012,
        validation_end_month=202112,
        expansion_weight=4.25,
        definition=CureDefinition(
            consecutive_current_months=3,
            outcome_horizon_months=12,
        ),
    )
    indexed = episodes.set_index(["split", "loan_id"])

    assert indexed.loc[("development", "train_cure"), "cured"] == 1
    assert indexed.loc[("development", "train_cure"), "cure_confirmation_month"] == (
        pd.Period("2020-04", freq="M")
    )
    assert indexed.loc[("development", "train_default"), "cured"] == 0
    assert indexed.loc[("validation", "validation_cure"), "cured"] == 1
    assert pd.isna(indexed.loc[("validation", "validation_rpl"), "cured"])
    assert episodes["sample_weight"].eq(4.25).all()
    assert diagnostics["definition"]["sustained_current_months"] == 3
    assert diagnostics["validation_observed_targets"] == 1


def test_realized_lgd_uses_disclosed_loss_and_release47_additive_signs() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["default15", "default02", "rpl"],
            "vintage": ["2005Q1"] * 3,
            "reporting_month": [201001, 201002, 201003],
            "state": [DEFAULTED, DEFAULTED, CENSORED_RPL],
            "zero_balance_code": ["15", "02", "16"],
            "actual_loss": [40.0, "", ""],
            "zero_balance_removal_upb": [100.0, 80.0, ""],
            "net_sales_proceeds": [-70.0, "", ""],
            "delinquent_accrued_interest": [5.0, "", ""],
            "total_expenses": [10.0, "", ""],
            "mi_recoveries": [-5.0, "", ""],
            "non_mi_recoveries": [0.0, "", ""],
        }
    )

    losses = extract_realized_lgd_rows(
        panel,
        development_end_month=201112,
        validation_end_month=201212,
        expansion_weight=10.0,
    ).set_index("loan_id")

    assert set(losses.index) == {"default15", "default02"}
    assert losses.loc["default15", "release47_status"] == "reconciled"
    assert losses.loc["default15", "release47_recomputed_loss"] == pytest.approx(40)
    assert losses.loc["default15", "realized_lgd_raw"] == pytest.approx(0.4)
    assert losses.loc["default15", "realized_lgd_model"] == pytest.approx(0.4)
    assert bool(losses.loc["default15", "resolved_default"]) is True
    assert losses.loc["default02", "release47_status"] == "actual_loss_missing"
    assert pd.isna(losses.loc["default02", "realized_lgd_raw"])


def _snapshot_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["loan-1", "paid"],
            "vintage": ["2005Q1", "2005Q1"],
            "reporting_month": [201212, 201212],
            "state": [CURRENT, "PAID_OFF"],
            "current_actual_upb": [100.0, 0.0],
            "current_interest_bearing_upb": [80.0, 0.0],
            "current_non_interest_bearing_upb": [20.0, 0.0],
            "delinquent_accrued_interest": [5.0, 0.0],
            "current_interest_rate": [6.0, 6.0],
            "remaining_months_to_legal_maturity": [12, 12],
            "orig_classic_fico": [720, 720],
            "orig_original_ltv": [75, 75],
            "orig_original_dti": [35, 35],
            "orig_original_interest_rate": [6.0, 6.0],
            "orig_original_upb": [120.0, 120.0],
            "orig_occupancy_status": ["P", "P"],
            "orig_property_type": ["SF", "SF"],
            "orig_loan_purpose": ["P", "P"],
            "loan_age": [24, 24],
            "estimated_ltv": [70, 70],
        }
    )


def test_snapshot_ead_components_and_amortizing_projection_are_auditable() -> None:
    snapshot = prepare_snapshot_inputs(
        _snapshot_panel(), snapshot_month=201212, expansion_weight=3.0
    )

    assert snapshot["loan_id"].tolist() == ["loan-1"]
    assert snapshot.loc[0, "ead"] == pytest.approx(105.0)
    assert snapshot.loc[0, "effective_annual_rate"] == pytest.approx(0.06)
    assert snapshot.loc[0, "ead_balance_basis"] == "component_fields"
    assert snapshot.loc[0, "fico_segment"] == "700 to 739"
    projection = project_snapshot_ead_sample(
        snapshot,
        maximum_loans=1,
        maximum_projection_months=12,
        seed=7,
    )
    assert projection.iloc[0]["default_ead"] == pytest.approx(105.0)
    assert projection.iloc[-1]["interest_bearing_upb"] == 0.0
    assert projection.iloc[-1]["default_ead"] == pytest.approx(25.0)


def test_reporting_snapshot_includes_default_and_uses_removal_upb_for_ead() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["active", "defaulted"],
            "vintage": ["2005Q1", "2005Q1"],
            "reporting_month": [201212, 201212],
            "state": [CURRENT, DEFAULTED],
            "current_actual_upb": [100.0, 0.0],
            "current_interest_bearing_upb": [100.0, 0.0],
            "current_non_interest_bearing_upb": [0.0, 0.0],
            "delinquent_accrued_interest": [0.0, 5.0],
            "zero_balance_removal_upb": [np.nan, 80.0],
        }
    )

    snapshot = prepare_snapshot_inputs(
        panel, snapshot_month=201212, expansion_weight=2.0
    ).set_index("loan_id")

    assert set(snapshot.index) == {"active", "defaulted"}
    assert snapshot.loc["defaulted", "ead"] == pytest.approx(85.0)
    assert snapshot.loc["defaulted", "ead_balance_basis"] == (
        "zero_balance_removal_upb_default"
    )


def test_cure_probability_is_operational_in_expected_lgd() -> None:
    inputs = pd.DataFrame(
        {
            "state": [CURRENT, DPD_30, "90_PLUS", DEFAULTED],
            "conditional_lgd": [0.5, 0.5, 0.5, 0.5],
            "cure_probability": [np.nan, 0.2, 0.4, 0.0],
        }
    )

    result = apply_cure_adjusted_lgd(inputs)

    np.testing.assert_allclose(result["expected_lgd"], [0.5, 0.4, 0.3, 0.5])
    assert result["cure_adjustment_applicable"].tolist() == [False, True, True, False]
    assert "cure_probability" in result.loc[1, "lgd_formula"]
    assert "terminal default" in result.loc[3, "lgd_formula"]


def test_out_of_time_ead_validation_uses_adjacent_unmodified_months() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["clean", "clean", "modified", "modified"],
            "reporting_month": [202101, 202102, 202101, 202102],
            "state": [CURRENT, CURRENT, CURRENT, CURRENT],
            "current_actual_upb": [100, 90, 100, 90],
            "current_interest_bearing_upb": [100, 90, 100, 90],
            "current_non_interest_bearing_upb": [0, 0, 0, 0],
            "current_interest_rate": [0, 0, 0, 0],
            "remaining_months_to_legal_maturity": [10, 9, 10, 9],
            "modification_flag": ["", "", "Y", "Y"],
        }
    )

    validation = build_ead_validation_rows(
        panel,
        validation_start_month=202101,
        validation_end_month=202112,
        expansion_weight=2.0,
    )
    metrics = evaluate_ead_validation(validation)

    assert validation["loan_id"].tolist() == ["clean"]
    assert validation.loc[0, "scheduled_next_balance"] == pytest.approx(90)
    assert validation.loc[0, "observed_next_balance"] == pytest.approx(90)
    assert metrics["rows"] == 1
    assert metrics["rmse"] == pytest.approx(0)


def test_validation_metric_helpers_report_weighted_out_of_time_fit() -> None:
    train_cure = pd.DataFrame(
        {
            "x": np.linspace(-2, 2, 80),
            "cured": np.r_[np.zeros(40), np.ones(40)],
            "sample_weight": np.where(np.arange(80) % 2, 2.0, 1.0),
        }
    )
    cure_model = fit_weighted_cure_model(
        train_cure,
        numeric_features=["x"],
        weight_col="sample_weight",
    )
    cure_metrics, cure_predictions, cure_calibration = evaluate_cure_predictions(
        cure_model, train_cure.iloc[::2].copy()
    )

    assert cure_metrics["rows"] == 40
    assert cure_metrics["roc_auc"] is not None
    assert cure_metrics["roc_auc"] > 0.9
    assert cure_predictions["predicted_cure_probability"].between(0, 1).all()
    assert not cure_calibration.empty

    severity_data = pd.DataFrame(
        {
            "x": np.linspace(-2, 2, 80),
            "realized_lgd_model": np.linspace(0.1, 0.9, 80),
            "resolved_default": True,
            "sample_weight": 1.0,
        }
    )
    severity_model = fit_conditional_severity_model(
        severity_data,
        numeric_features=["x"],
        condition_col="resolved_default",
        condition_value=True,
        weight_col="sample_weight",
    )
    severity_metrics, severity_predictions, severity_calibration = (
        evaluate_severity_predictions(severity_model, severity_data.iloc[::2].copy())
    )

    assert severity_metrics["rows"] == 40
    assert severity_metrics["rmse"] < 0.1
    assert severity_predictions["predicted_conditional_lgd"].between(0, 1).all()
    assert not severity_calibration.empty


def test_stable_sampling_and_run_signature_do_not_depend_on_row_order() -> None:
    data = pd.DataFrame({"loan_id": [f"L{i}" for i in range(20)], "value": range(20)})
    first = stable_row_sample(data, 5, id_columns=["loan_id"], seed=9)
    second = stable_row_sample(
        data.sample(frac=1, random_state=2),
        5,
        id_columns=["loan_id"],
        seed=9,
    )
    assert set(first["loan_id"]) == set(second["loan_id"])

    settings = Phase3Settings(
        vintages=("2005Q1",),
        development_end_month=201112,
        validation_end_month=201212,
        snapshot_month=201212,
        cure_consecutive_current_months=3,
        cure_horizon_months=12,
        maximum_iterations=1000,
        random_seed=42,
        maximum_projection_months=360,
        projection_sample_loans=250,
        reuse_completed_quarters=True,
    )
    sources = [{"vintage": "2005Q1", "panel_rows": 10, "expansion_weight": 2.0}]
    assert phase3_run_signature(settings, sources) == phase3_run_signature(
        settings, sources
    )

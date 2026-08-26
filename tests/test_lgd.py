"""Tests for cure episodes, cure modeling, and realized loss severity."""

import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.lgd import (
    CureDefinition,
    add_realized_lgd,
    build_delinquency_episodes,
    calculate_realized_lgd,
    fit_conditional_severity_model,
    fit_weighted_cure_model,
)
from ifrs9_ecl.states import CURRENT, DEFAULTED, DPD_30, DPD_60, DPD_90_PLUS


def test_builds_sustained_cure_default_and_censored_episode_outcomes() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [
                "cure",
                "cure",
                "cure",
                "cure",
                "default",
                "default",
                "short",
                "short",
                "horizon",
                "horizon",
                "horizon",
                "gap",
                "gap",
            ],
            "reporting_month": [
                202001,
                202002,
                202003,
                202004,
                202001,
                202002,
                202001,
                202002,
                202001,
                202002,
                202003,
                202001,
                202003,
            ],
            "state": [
                CURRENT,
                DPD_30,
                CURRENT,
                CURRENT,
                DPD_60,
                DEFAULTED,
                DPD_30,
                DPD_60,
                DPD_30,
                DPD_60,
                DPD_90_PLUS,
                DPD_30,
                DPD_60,
            ],
            "fico": [700] * 13,
        }
    )
    definition = CureDefinition(
        consecutive_current_months=2, outcome_horizon_months=2
    )

    result = build_delinquency_episodes(
        panel, definition=definition, feature_cols=["fico"]
    )
    episodes = result.episodes.set_index("loan_id")

    assert episodes.loc["cure", "event_type"] == "cured"
    assert episodes.loc["cure", "event_month"] == pd.Period("2020-03", freq="M")
    assert episodes.loc["cure", "cure_confirmation_month"] == pd.Period(
        "2020-04", freq="M"
    )
    assert episodes.loc["cure", "cured"] == 1
    assert episodes.loc["default", "event_type"] == "defaulted"
    assert episodes.loc["default", "cured"] == 0
    assert pd.isna(episodes.loc["short", "cured"])
    assert episodes.loc["horizon", "cured"] == 0
    assert episodes.loc["horizon", "outcome_reason"] == "not_cured_within_horizon"
    gap_episodes = result.episodes[result.episodes["loan_id"] == "gap"]
    assert gap_episodes.iloc[0]["event_type"] == "gap"
    assert gap_episodes["cured"].isna().all()
    assert episodes.loc["cure", "fico"] == 700
    assert result.diagnostics.episodes == 6
    assert result.diagnostics.gaps_encountered == 1


def test_failed_one_month_cure_stays_in_same_episode_until_sustained() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1, 1, 1],
            "reporting_month": [202001, 202002, 202003, 202004, 202005],
            "state": [DPD_30, CURRENT, DPD_60, CURRENT, CURRENT],
        }
    )

    episodes = build_delinquency_episodes(
        panel,
        definition=CureDefinition(
            consecutive_current_months=2, outcome_horizon_months=None
        ),
    ).episodes

    assert len(episodes) == 1
    assert episodes.loc[0, "delinquent_months"] == 2
    assert episodes.loc[0, "worst_state"] == DPD_60
    assert episodes.loc[0, "event_month"] == pd.Period("2020-04", freq="M")
    assert episodes.loc[0, "cured"] == 1


def test_episode_duplicate_rules_do_not_create_unsupported_paths() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1, 1, 1],
            "reporting_month": [202001, 202001, 202002, 202002, 202003],
            "state": [DPD_30, DPD_30, CURRENT, DPD_60, CURRENT],
        }
    )
    before = panel.copy(deep=True)

    result = build_delinquency_episodes(panel)

    assert result.diagnostics.identical_duplicate_rows_collapsed == 1
    assert result.diagnostics.conflicting_loan_months_dropped == 1
    assert result.episodes.loc[0, "event_type"] == "gap"
    pd.testing.assert_frame_equal(panel, before)


@pytest.mark.parametrize(
    ("loss", "upb", "raw", "status"),
    [
        (25.0, 100.0, 0.25, "in_unit_interval"),
        (-10.0, 100.0, -0.10, "gain_below_zero"),
        (125.0, 100.0, 1.25, "loss_above_one"),
    ],
)
def test_realized_lgd_preserves_raw_economics(
    loss: float, upb: float, raw: float, status: str
) -> None:
    result = calculate_realized_lgd(loss, upb)

    assert result.raw_lgd == pytest.approx(raw)
    assert result.modeling_lgd == pytest.approx(raw)
    assert result.status == status
    assert result.is_gain is (loss < 0)


def test_realized_lgd_treatments_are_explicit_and_keep_audit_flags() -> None:
    clipped_gain = calculate_realized_lgd(-10, 100, out_of_range="clip")
    excluded_high = calculate_realized_lgd(125, 100, out_of_range="exclude")

    assert clipped_gain.raw_lgd == -0.1
    assert clipped_gain.modeling_lgd == 0.0
    assert clipped_gain.below_zero is True
    assert excluded_high.raw_lgd == 1.25
    assert excluded_high.modeling_lgd is None
    assert excluded_high.eligible_for_model is False
    with pytest.raises(ValueError, match="outside the unit interval"):
        calculate_realized_lgd(-1, 100, out_of_range="error")


def test_realized_lgd_missing_or_nonpositive_denominator_is_not_divided() -> None:
    missing = calculate_realized_lgd(None, 100)
    zero = calculate_realized_lgd(10, 0)

    assert missing.status == "actual_loss_missing"
    assert missing.raw_lgd is None
    assert zero.status == "nonpositive_removal_upb"
    assert zero.raw_lgd is None


def test_add_realized_lgd_returns_copy_with_raw_and_model_columns() -> None:
    data = pd.DataFrame(
        {"actual_loss": [20, -10, 120], "zero_balance_removal_upb": [100, 100, 100]}
    )
    before = data.copy(deep=True)

    result = add_realized_lgd(data, out_of_range="exclude")

    assert result["realized_lgd_raw"].tolist() == [0.2, -0.1, 1.2]
    assert result["realized_lgd_model"].iloc[0] == 0.2
    assert result["realized_lgd_model"].iloc[1:].isna().all()
    assert result["realized_lgd_outside_unit_interval"].tolist() == [False, True, True]
    pd.testing.assert_frame_equal(data, before)


def test_weighted_cure_model_preprocesses_mixed_features_and_reports_fit() -> None:
    rng = np.random.default_rng(42)
    n = 300
    fico = rng.normal(700, 45, n)
    channel = rng.choice(["R", "B", "C"], n)
    linear = -0.4 + (fico - 700) / 55 + (channel == "R") * 0.7
    probability = 1 / (1 + np.exp(-linear))
    cured = rng.binomial(1, probability).astype("float64")
    cured[0] = np.nan
    data = pd.DataFrame(
        {
            "fico": fico,
            "channel": channel,
            "cured": cured,
            "weight": np.where(channel == "R", 2.0, 1.0),
        }
    )

    model = fit_weighted_cure_model(
        data,
        numeric_features=["fico"],
        categorical_features=["channel"],
        weight_col="weight",
    )
    prediction = model.predict_probability(
        pd.DataFrame({"fico": [650, 750], "channel": ["NEW", "R"]})
    )

    assert model.diagnostics.converged is True
    assert model.diagnostics.fitted_rows == n - 1
    assert model.diagnostics.excluded_missing_target_rows == 1
    assert 0 < model.diagnostics.roc_auc <= 1
    assert np.all((prediction > 0) & (prediction < 1))
    assert prediction.shape == (2,)


def test_weighted_cure_model_rejects_invalid_weights_and_one_class_target() -> None:
    one_class = pd.DataFrame({"x": [1, 2], "cured": [1, 1]})
    with pytest.raises(ValueError, match="both zero and one"):
        fit_weighted_cure_model(one_class, numeric_features=["x"])

    negative_weight = pd.DataFrame(
        {"x": [1, 2], "cured": [0, 1], "weight": [1, -1]}
    )
    with pytest.raises(ValueError, match="nonnegative"):
        fit_weighted_cure_model(
            negative_weight, numeric_features=["x"], weight_col="weight"
        )


def test_weighted_cure_model_fails_loudly_when_solver_does_not_converge() -> None:
    rng = np.random.default_rng(123)
    features = rng.normal(size=(400, 6))
    target = rng.binomial(1, 1 / (1 + np.exp(-4 * features[:, 0])))
    data = pd.DataFrame(features, columns=[f"x{index}" for index in range(6)])
    data["cured"] = target

    with pytest.raises(RuntimeError, match="did not converge"):
        fit_weighted_cure_model(
            data,
            numeric_features=[f"x{index}" for index in range(6)],
            max_iter=1,
            tolerance=1e-12,
        )

    diagnostic_fit = fit_weighted_cure_model(
        data,
        numeric_features=[f"x{index}" for index in range(6)],
        max_iter=1,
        tolerance=1e-12,
        raise_on_nonconvergence=False,
    )
    assert diagnostic_fit.diagnostics.converged is False
    assert diagnostic_fit.diagnostics.iterations == 1
    assert diagnostic_fit.diagnostics.warnings


def test_conditional_severity_model_filters_defaults_and_has_fit_diagnostics() -> None:
    rng = np.random.default_rng(7)
    n = 250
    ltv = rng.uniform(40, 130, n)
    property_type = rng.choice(["SF", "CO"], n)
    mean = 1 / (1 + np.exp(-(-2.0 + ltv / 60 + (property_type == "CO") * 0.3)))
    severity = np.clip(mean + rng.normal(0, 0.06, n), 0.01, 0.99)
    data = pd.DataFrame(
        {
            "ltv": ltv,
            "property_type": property_type,
            "realized_lgd_model": severity,
            "resolved_default": np.r_[np.ones(n - 10), np.zeros(10)],
            "weight": rng.uniform(0.5, 2.0, n),
        }
    )

    model = fit_conditional_severity_model(
        data,
        numeric_features=["ltv"],
        categorical_features=["property_type"],
        condition_col="resolved_default",
        weight_col="weight",
    )
    with pytest.warns(UserWarning, match="unknown categories"):
        prediction = model.predict(
            pd.DataFrame({"ltv": [60, 120], "property_type": ["NEW", "CO"]})
        )

    assert model.diagnostics.converged is True
    assert model.diagnostics.condition_eligible_rows == n - 10
    assert model.diagnostics.excluded_by_condition_rows == 10
    assert model.diagnostics.fitted_rows == n - 10
    assert np.isfinite(model.diagnostics.aic)
    assert model.diagnostics.weighted_rmse >= 0
    assert np.all((prediction >= 0) & (prediction <= 1))
    assert prediction[1] > prediction[0]


def test_conditional_severity_out_of_range_treatment_is_diagnosed() -> None:
    data = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4],
            "realized_lgd_model": [-0.1, 0.2, 0.4, 0.8, 1.2],
        }
    )

    excluded = fit_conditional_severity_model(
        data,
        numeric_features=["x"],
        out_of_range="exclude",
    )
    clipped = fit_conditional_severity_model(
        data,
        numeric_features=["x"],
        out_of_range="clip",
    )

    assert excluded.diagnostics.outside_unit_interval_rows == 2
    assert excluded.diagnostics.excluded_outside_unit_interval_rows == 2
    assert excluded.diagnostics.fitted_rows == 3
    assert clipped.diagnostics.clipped_outside_unit_interval_rows == 2
    assert clipped.diagnostics.fitted_rows == 5
    with pytest.raises(ValueError, match="outside the unit interval"):
        fit_conditional_severity_model(
            data, numeric_features=["x"], out_of_range="error"
        )


def test_conditional_severity_nonconvergence_is_not_silent() -> None:
    x = np.linspace(-3, 3, 200)
    data = pd.DataFrame(
        {"x": x, "realized_lgd_model": 1 / (1 + np.exp(-10 * x))}
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        fit_conditional_severity_model(
            data,
            numeric_features=["x"],
            max_iter=1,
            tolerance=1e-12,
        )

    diagnostic_fit = fit_conditional_severity_model(
        data,
        numeric_features=["x"],
        max_iter=1,
        tolerance=1e-12,
        raise_on_nonconvergence=False,
    )
    assert diagnostic_fit.diagnostics.converged is False
    assert diagnostic_fit.diagnostics.iterations == 1

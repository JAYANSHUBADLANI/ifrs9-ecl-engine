import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.pd_explain import explain_hazard
from ifrs9_ecl.pd_model import DiscreteTimeHazardModel


@pytest.fixture
def hazard_training_data():
    ages = np.tile(np.arange(1, 13), 8)
    regions = np.repeat(["north", "south"], len(ages) // 2)
    ltv = 55.0 + (ages * 3.0) + np.tile([0.0, 8.0], len(ages) // 2)
    events = ((ages >= 9) & (np.arange(len(ages)) % 3 != 0)).astype(int)
    frame = pd.DataFrame(
        {
            "loan_age_months": ages,
            "ltv": ltv,
            "region": regions,
        }
    )
    return frame, events


@pytest.fixture
def fitted_model(hazard_training_data):
    frame, events = hazard_training_data
    model = DiscreteTimeHazardModel(
        numeric_features=["loan_age_months", "ltv"],
        categorical_features=["region"],
    )
    model.fit(frame, events, sample_weight=np.ones(len(frame)))
    return model, frame


def test_explanation_is_additive(fitted_model):
    model, frame = fitted_model
    explanations = explain_hazard(model, frame.iloc[:5], background=frame)
    for explanation in explanations:
        assert explanation.check_additivity()


def test_hazard_in_explanation_matches_model_prediction(fitted_model):
    model, frame = fitted_model
    sample = frame.iloc[:6]
    explanations = explain_hazard(model, sample, background=frame)
    direct = model.predict_hazard(sample)
    for explanation, hazard in zip(explanations, direct):
        assert explanation.hazard == pytest.approx(float(hazard), abs=1e-8)


def test_every_fitted_feature_has_exactly_one_contribution(fitted_model):
    model, frame = fitted_model
    explanations = explain_hazard(model, frame.iloc[:1], background=frame)
    features = {c.feature for c in explanations[0].contributions}
    assert features == set(model.feature_columns)


def test_numeric_contribution_at_the_background_mean_age_is_near_zero(fitted_model):
    # loan_age_months is standardized on the fit time training mean, and the
    # background here is exactly that training frame, so a loan sitting at
    # the mean age should show close to zero contribution from that feature,
    # not merely a small one: this is the case the closed form is exact for
    # without needing any background at all for numeric features.
    model, frame = fitted_model
    mean_age = frame["loan_age_months"].mean()
    at_mean = frame.iloc[(frame["loan_age_months"] - mean_age).abs().idxmin() :][
        :1
    ].copy()
    at_mean["loan_age_months"] = mean_age
    explanation = explain_hazard(model, at_mean, background=frame)[0]
    age_contribution = next(
        c for c in explanation.contributions if c.feature == "loan_age_months"
    )
    assert abs(age_contribution.contribution) < 0.05


def test_ranked_orders_by_absolute_contribution_descending(fitted_model):
    model, frame = fitted_model
    explanation = explain_hazard(model, frame.iloc[:1], background=frame)[0]
    ranked = explanation.ranked()
    magnitudes = [abs(c.contribution) for c in ranked]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_empty_background_is_rejected(fitted_model):
    model, frame = fitted_model
    with pytest.raises(ValueError, match="background must contain"):
        explain_hazard(model, frame.iloc[:1], background=frame.iloc[:0])


def test_as_dict_round_trips_the_ranked_contributions(fitted_model):
    model, frame = fitted_model
    explanation = explain_hazard(model, frame.iloc[:1], background=frame)[0]
    payload = explanation.as_dict()
    assert payload["hazard"] == pytest.approx(explanation.hazard)
    assert len(payload["contributions"]) == len(model.feature_columns)
    contribution_order = [c["contribution"] for c in payload["contributions"]]
    assert [abs(v) for v in contribution_order] == sorted(
        (abs(v) for v in contribution_order), reverse=True
    )

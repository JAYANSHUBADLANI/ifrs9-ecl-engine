import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from ifrs9_ecl.pd_model import (
    MODEL_ARTIFACT_TYPE,
    MODEL_ARTIFACT_VERSION,
    DiscreteTimeHazardModel,
)


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


def _new_model(**kwargs):
    return DiscreteTimeHazardModel(
        numeric_features=["loan_age_months", "ltv"],
        categorical_features=["region"],
        **kwargs,
    )


def test_hazard_model_fits_mixed_features_with_explicit_weights(hazard_training_data):
    frame, events = hazard_training_data
    weights = np.where(events == 1, 4.0, 0.5)
    model = _new_model().fit(frame, events, sample_weight=weights)

    predictions = model.predict_hazard(frame.iloc[:5])
    assert predictions.shape == (5,)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    assert model.fit_diagnostics_.sample_weight_provided
    assert model.fit_diagnostics_.weight_sum == pytest.approx(weights.sum())
    assert model.fit_diagnostics_.weighted_events == pytest.approx(weights[events == 1].sum())
    assert model.fit_diagnostics_.transformed_features == 4
    assert model.converged_
    assert all(iterations < model.max_iter for iterations in model.n_iter_)


def test_unseen_category_is_handled_at_prediction(hazard_training_data):
    frame, events = hazard_training_data
    model = _new_model().fit(frame, events, sample_weight=np.ones(len(frame)))
    scoring = pd.DataFrame(
        {
            "loan_age_months": [4, 8],
            "ltv": [70.0, 90.0],
            "region": ["new_region", None],
        }
    )

    hazards = model.predict_hazard(scoring)
    assert hazards.shape == (2,)
    assert np.isfinite(hazards).all()


def test_projection_advances_loan_age_for_every_month(hazard_training_data):
    frame, events = hazard_training_data
    model = _new_model().fit(frame, events, sample_weight=np.ones(len(frame)))
    loan = pd.DataFrame(
        {"loan_age_months": [5], "ltv": [80.0], "region": ["north"]}
    )
    direct = pd.DataFrame(
        {
            "loan_age_months": [6, 7, 8],
            "ltv": [80.0, 80.0, 80.0],
            "region": ["north", "north", "north"],
        }
    )

    projected = model.predict_hazard_curve(loan, 3)
    np.testing.assert_allclose(projected[0], model.predict_hazard(direct))


def test_projection_uses_row_positions_when_input_index_is_duplicated(
    hazard_training_data,
):
    frame, events = hazard_training_data
    model = _new_model().fit(frame, events, sample_weight=np.ones(len(frame)))
    loans = frame.iloc[:2].copy()
    loans.index = [7, 7]

    projected = model.predict_hazard_curve(loans, 4)
    assert projected.shape == (2, 4)


def test_projected_pd_curves_reconcile_to_hazards(hazard_training_data):
    frame, events = hazard_training_data
    model = _new_model().fit(frame, events, sample_weight=np.ones(len(frame)))
    curves = model.predict_pd_curves(frame.iloc[:3], 12)

    assert curves.conditional_hazard.shape == (3, 12)
    assert np.all(np.diff(curves.cumulative_pd, axis=1) >= -1e-15)
    np.testing.assert_allclose(curves.marginal_pd.sum(axis=1), curves.cumulative_pd[:, -1])
    np.testing.assert_allclose(curves.survival + curves.cumulative_pd, 1.0)


def test_convergence_status_checks_n_iter_against_iteration_cap(hazard_training_data):
    frame, events = hazard_training_data
    model = _new_model(max_iter=1)

    with pytest.warns(ConvergenceWarning):
        model.fit(frame, events, sample_weight=np.ones(len(frame)))

    assert model.n_iter_ == (1,)
    assert not model.converged_
    assert not model.check_convergence()
    with pytest.raises(RuntimeError, match="did not converge"):
        model.check_convergence(raise_on_failure=True)


def test_versioned_serialization_preserves_predictions_and_diagnostics(
    hazard_training_data, tmp_path
):
    frame, events = hazard_training_data
    model = _new_model().fit(frame, events, sample_weight=np.ones(len(frame)))
    path = model.save(tmp_path / "hazard_model.pkl")

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["artifact_type"] == MODEL_ARTIFACT_TYPE
    assert payload["artifact_version"] == MODEL_ARTIFACT_VERSION
    assert tuple(payload["model_config"]["numeric_features"]) == model.numeric_features

    loaded = DiscreteTimeHazardModel.load(path)
    np.testing.assert_array_equal(
        loaded.predict_hazard(frame.iloc[:10]),
        model.predict_hazard(frame.iloc[:10]),
    )
    assert loaded.fit_diagnostics_ == model.fit_diagnostics_
    assert loaded.serialization_metadata() == model.serialization_metadata()


@pytest.mark.parametrize(
    ("y", "weights", "message"),
    [
        ([0, 1], [1.0], "one value"),
        ([0, 1], [1.0, -1.0], "nonnegative"),
        ([0, 1], [0.0, 0.0], "positive total"),
        ([0, 1], [1.0, 0.0], "both target classes"),
    ],
)
def test_invalid_sample_weights_fail_loudly(y, weights, message):
    frame = pd.DataFrame(
        {
            "loan_age_months": [1, 2],
            "ltv": [70.0, 80.0],
            "region": ["north", "south"],
        }
    )
    with pytest.raises(ValueError, match=message):
        _new_model().fit(frame, y, sample_weight=weights)


def test_missing_features_and_invalid_projection_age_fail_loudly(hazard_training_data):
    frame, events = hazard_training_data
    model = _new_model().fit(frame, events, sample_weight=np.ones(len(frame)))

    with pytest.raises(KeyError, match="missing model features"):
        model.predict_hazard(frame.drop(columns="ltv"))
    invalid_age = frame.iloc[:1].copy()
    invalid_age["loan_age_months"] = np.nan
    with pytest.raises(ValueError, match="loan ages"):
        model.predict_hazard_curve(invalid_age, 12)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict_hazard_curve(frame.iloc[:1], 0)


def test_unfitted_model_cannot_score_or_serialize(tmp_path):
    model = _new_model()
    frame = pd.DataFrame(
        {"loan_age_months": [1], "ltv": [70.0], "region": ["north"]}
    )
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_hazard(frame)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.save(tmp_path / "model.pkl")

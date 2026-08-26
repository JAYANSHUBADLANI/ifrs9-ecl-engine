import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.survival import (
    conditional_hazards_to_pd,
    estimate_kaplan_meier,
    hazards_to_pd_curves,
    kaplan_meier,
)


def test_kaplan_meier_product_limit_with_tied_event_and_censoring():
    result = kaplan_meier(
        durations=[1, 2, 2, 3],
        event_observed=[1, 1, 0, 1],
    )

    np.testing.assert_allclose(result.curve["at_risk"], [4.0, 3.0, 1.0])
    np.testing.assert_allclose(result.curve["events"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.curve["censored"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(result.curve["survival"], [0.75, 0.5, 0.0])
    np.testing.assert_allclose(
        result.curve["cumulative_incidence"], [0.25, 0.5, 1.0]
    )
    assert result.diagnostics.events == 3
    assert result.diagnostics.censored == 1
    assert result.diagnostics.censoring_rate == 0.25
    assert result.diagnostics.tied_times == 1
    assert result.diagnostics.mixed_event_censor_times == 1
    assert result.diagnostics.median_survival_time == 2.0


def test_kaplan_meier_all_censored_stays_at_one():
    result = estimate_kaplan_meier([1, 2, 4], [0, 0, 0])

    np.testing.assert_allclose(result.table["survival"], 1.0)
    np.testing.assert_allclose(result.table["hazard"], 0.0)
    assert result.diagnostics.median_survival_time is None
    assert result.diagnostics.censoring_rate == 1.0
    assert result.survival_at(0.0) == 1.0
    assert result.survival_at(100.0) == 1.0


def test_kaplan_meier_uses_explicit_weights_and_audits_zero_weights():
    result = kaplan_meier(
        durations=[1, 1, 2],
        event_observed=[1, 0, 1],
        sample_weight=[2.0, 1.0, 0.0],
    )

    assert result.curve["time"].tolist() == [1.0]
    assert result.curve.loc[0, "at_risk"] == 3.0
    assert result.curve.loc[0, "survival"] == pytest.approx(1.0 / 3.0)
    assert result.diagnostics.observations == 3
    assert result.diagnostics.zero_weight_observations == 1
    assert result.diagnostics.weighted_events == 2.0
    assert result.diagnostics.weighted_censored == 1.0


@pytest.mark.parametrize(
    ("durations", "events", "weights", "message"),
    [
        ([], [], None, "at least one"),
        ([1, -1], [0, 1], None, "nonnegative"),
        ([1, 2], [0], None, "same length"),
        ([1, 2], [0, 2], None, "zero and one"),
        ([1, 2], [0, 1], [1], "one value"),
        ([1, 2], [0, 1], [0, 0], "positive total"),
    ],
)
def test_kaplan_meier_rejects_invalid_inputs(durations, events, weights, message):
    with pytest.raises(ValueError, match=message):
        kaplan_meier(durations, events, sample_weight=weights)


def test_kaplan_meier_does_not_mutate_series_inputs():
    durations = pd.Series([1.0, 2.0, 3.0], name="duration")
    events = pd.Series([1, 0, 1], name="event")
    before_durations = durations.copy(deep=True)
    before_events = events.copy(deep=True)

    kaplan_meier(durations, events)

    pd.testing.assert_series_equal(durations, before_durations)
    pd.testing.assert_series_equal(events, before_events)


def test_hazards_convert_to_reconciled_pd_curves():
    curves = conditional_hazards_to_pd([0.10, 0.20, 0.50])

    np.testing.assert_allclose(curves.survival, [0.90, 0.72, 0.36])
    np.testing.assert_allclose(curves.marginal_pd, [0.10, 0.18, 0.36])
    np.testing.assert_allclose(curves.cumulative_pd, [0.10, 0.28, 0.64])
    np.testing.assert_allclose(np.cumsum(curves.marginal_pd), curves.cumulative_pd)
    assert curves.to_frame()["month"].tolist() == [1, 2, 3]


def test_hazard_conversion_supports_batched_curves_and_an_explicit_axis():
    hazards = np.array([[0.1, 0.2], [0.4, 0.5], [0.0, 1.0]])
    curves = hazards_to_pd_curves(hazards, axis=0)

    assert curves.survival.shape == hazards.shape
    np.testing.assert_allclose(curves.cumulative_pd[-1], [0.46, 1.0])
    np.testing.assert_allclose(curves.marginal_pd.sum(axis=0), curves.cumulative_pd[-1])
    with pytest.raises(ValueError, match="one-dimensional"):
        curves.to_frame()


@pytest.mark.parametrize("hazards", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_hazard_conversion_rejects_invalid_probabilities(hazards):
    with pytest.raises(ValueError):
        conditional_hazards_to_pd(hazards)

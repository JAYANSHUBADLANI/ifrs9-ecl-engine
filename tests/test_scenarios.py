"""Tests for conditional-hazard macro scenario overlays."""

import numpy as np
import pytest

from ifrs9_ecl.scenarios import (
    DEFAULT_SCENARIOS,
    Scenario,
    apply_hazard_shift,
    calculate_scenario_ecl,
    conditional_hazards_from_marginal,
    marginal_pd_from_hazards,
    normalize_scenarios,
)


def test_default_scenario_set_has_three_named_normalized_paths() -> None:
    assert {scenario.name for scenario in DEFAULT_SCENARIOS} == {
        "base",
        "upside",
        "downside",
    }
    assert sum(scenario.weight for scenario in DEFAULT_SCENARIOS) == pytest.approx(1.0)
    assert next(s for s in DEFAULT_SCENARIOS if s.name == "upside").hazard_shift < 0
    assert next(s for s in DEFAULT_SCENARIOS if s.name == "downside").hazard_shift > 0


def test_scenario_weights_are_probability_normalized_without_mutating_inputs() -> None:
    scenarios = (
        Scenario("base", 6.0, 0.0),
        Scenario("upside", 2.0, -0.2),
        Scenario("downside", 2.0, 0.3),
    )

    normalized = normalize_scenarios(scenarios)

    assert [scenario.weight for scenario in normalized] == pytest.approx([0.6, 0.2, 0.2])
    assert [scenario.weight for scenario in scenarios] == [6.0, 2.0, 2.0]


def test_marginal_and_conditional_hazard_conversions_round_trip() -> None:
    marginal = np.asarray([0.10, 0.18, 0.072])

    hazards = conditional_hazards_from_marginal(marginal)
    rebuilt = marginal_pd_from_hazards(hazards)

    assert hazards.tolist() == pytest.approx([0.10, 0.20, 0.10])
    assert rebuilt.tolist() == pytest.approx(marginal)


def test_zero_hazard_shift_preserves_marginal_curve() -> None:
    marginal = np.asarray([0.01, 0.02, 0.03])

    shifted = apply_hazard_shift(marginal, 0.0)

    assert shifted.tolist() == pytest.approx(marginal)


def test_positive_shift_increases_cumulative_pd_and_negative_shift_reduces_it() -> None:
    marginal = np.asarray([0.01] * 24)

    upside = apply_hazard_shift(marginal, -0.5)
    downside = apply_hazard_shift(marginal, 0.5)

    assert upside.sum() < marginal.sum() < downside.sum()
    assert upside.sum() <= 1.0
    assert downside.sum() <= 1.0


def test_probability_weighted_ecl_reconciles_to_scenario_rows() -> None:
    scenarios = (
        Scenario("base", 60.0, 0.0),
        Scenario("upside", 20.0, -0.5),
        Scenario("downside", 20.0, 0.5),
    )
    result = calculate_scenario_ecl(
        [0.01] * 18,
        0.5,
        100.0,
        stage=2,
        scenarios=scenarios,
    )
    frame = result.to_frame().set_index("scenario")

    assert frame.loc["upside", "ecl"] < frame.loc["base", "ecl"]
    assert frame.loc["base", "ecl"] < frame.loc["downside", "ecl"]
    assert result.weighted_ecl == pytest.approx(frame["weighted_ecl"].sum())
    assert frame["weight"].sum() == pytest.approx(1.0)


def test_stage_1_scenario_ecl_still_uses_12_month_horizon() -> None:
    result = calculate_scenario_ecl(
        [0.01] * 24,
        [0.5] * 24,
        [100.0] * 24,
        stage=1,
    )

    assert {outcome.ecl_result.horizon_months for outcome in result.outcomes} == {12}


def test_stage_3_ecl_is_not_changed_by_macro_hazard_shift() -> None:
    result = calculate_scenario_ecl(
        None,
        None,
        None,
        stage=3,
        stage3_cash_shortfall=40.0,
    )

    assert [outcome.ecl for outcome in result.outcomes] == pytest.approx([40.0] * 3)
    assert result.weighted_ecl == pytest.approx(40.0)


@pytest.mark.parametrize(
    "scenarios",
    [
        (),
        (Scenario("base", 0.0), Scenario("downside", 0.0)),
        (Scenario("base", 1.0), Scenario("BASE", 1.0)),
    ],
)
def test_invalid_scenario_sets_are_rejected(scenarios: tuple[Scenario, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_scenarios(scenarios)


@pytest.mark.parametrize("hazards", [[], [-0.1], [1.1], [float("nan")]])
def test_invalid_conditional_hazards_are_rejected(hazards: list[float]) -> None:
    with pytest.raises((TypeError, ValueError), match="hazards"):
        marginal_pd_from_hazards(hazards)


def test_scenario_rejects_negative_weight_and_nonfinite_shift() -> None:
    with pytest.raises(ValueError, match="weight"):
        Scenario("bad", -0.1)
    with pytest.raises(ValueError, match="hazard_shift"):
        Scenario("bad", 1.0, float("inf"))

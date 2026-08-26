"""Tests for the transparent IFRS9 staging policy."""

import math

import pandas as pd
import pytest

from ifrs9_ecl.staging import (
    Stage,
    StagingPolicy,
    assess_stage,
    assign_stage,
    assign_stages,
    normalize_stage,
)


def test_performing_exposure_is_stage_1() -> None:
    decision = assess_stage(0.015, 0.01, days_past_due=0)

    assert decision.stage is Stage.STAGE_1
    assert decision.reason == "performing"
    assert decision.lifetime_pd_ratio == pytest.approx(1.5)
    assert assign_stage(0.015, 0.01) == 1


def test_relative_lifetime_pd_increase_triggers_stage_2_at_threshold() -> None:
    decision = assess_stage(0.02, 0.01, days_past_due=0)

    assert decision.stage is Stage.STAGE_2
    assert decision.reason == "relative_pd_sicr"
    assert decision.pd_sicr_triggered is True
    assert decision.delinquency_backstop_triggered is False


def test_30_dpd_backstop_triggers_stage_2_without_pd_deterioration() -> None:
    decision = assess_stage(0.005, 0.01, days_past_due=30)

    assert decision.stage is Stage.STAGE_2
    assert decision.reason == "30_dpd_backstop"
    assert decision.pd_sicr_triggered is False
    assert decision.delinquency_backstop_triggered is True


def test_90_dpd_and_default_flag_each_trigger_stage_3_with_precedence() -> None:
    by_dpd = assess_stage(0.005, 0.01, days_past_due=90)
    by_default = assess_stage(0.005, 0.01, days_past_due=0, is_default=True)

    assert by_dpd.stage is Stage.STAGE_3
    assert by_dpd.reason == "90_dpd_backstop"
    assert by_default.stage is Stage.STAGE_3
    assert by_default.reason == "default_flag"


def test_positive_pd_over_zero_origination_pd_is_an_infinite_sicr_ratio() -> None:
    decision = assess_stage(0.001, 0.0)
    no_change = assess_stage(0.0, 0.0)

    assert decision.stage is Stage.STAGE_2
    assert math.isinf(decision.lifetime_pd_ratio)
    assert no_change.stage is Stage.STAGE_1
    assert no_change.lifetime_pd_ratio == 1.0


def test_custom_policy_changes_relative_and_dpd_thresholds() -> None:
    policy = StagingPolicy(
        sicr_relative_threshold=3.0,
        stage2_dpd_backstop=45,
        stage3_dpd_backstop=120,
    )

    assert assess_stage(0.025, 0.01, days_past_due=44, policy=policy).stage == 1
    assert assess_stage(0.025, 0.01, days_past_due=45, policy=policy).stage == 2
    assert assess_stage(0.025, 0.01, days_past_due=120, policy=policy).stage == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sicr_relative_threshold": 1.0},
        {"stage2_dpd_backstop": 90, "stage3_dpd_backstop": 90},
        {"stage2_dpd_backstop": -1},
    ],
)
def test_invalid_policy_fails_loudly(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        StagingPolicy(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_lifetime_pd", 1.1),
        ("origination_lifetime_pd", -0.1),
        ("days_past_due", 1.5),
        ("is_default", 1),
    ],
)
def test_invalid_exposure_inputs_fail_loudly(field: str, value: object) -> None:
    inputs: dict[str, object] = {
        "current_lifetime_pd": 0.01,
        "origination_lifetime_pd": 0.01,
        "days_past_due": 0,
        "is_default": False,
    }
    inputs[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        assess_stage(**inputs)


def test_assign_stages_is_auditable_and_does_not_mutate_input() -> None:
    exposures = pd.DataFrame(
        {
            "current_lifetime_pd": [0.01, 0.03, 0.01, 0.01],
            "origination_lifetime_pd": [0.01, 0.01, 0.01, 0.01],
            "days_past_due": [0, 0, 30, 0],
            "is_default": [False, False, False, True],
        },
        index=[10, 11, 12, 13],
    )
    before = exposures.copy(deep=True)

    result = assign_stages(exposures)

    assert result["stage"].tolist() == [1, 2, 2, 3]
    assert result["staging_reason"].tolist() == [
        "performing",
        "relative_pd_sicr",
        "30_dpd_backstop",
        "default_flag",
    ]
    assert result["stage"].dtype.name == "int8"
    pd.testing.assert_frame_equal(exposures, before)


def test_assign_stages_supports_empty_frame_and_custom_columns() -> None:
    exposures = pd.DataFrame(columns=["now", "at_start", "dpd", "bad"])

    result = assign_stages(
        exposures,
        current_pd_col="now",
        origination_pd_col="at_start",
        days_past_due_col="dpd",
        default_col="bad",
    )

    assert result.empty
    assert {"stage", "staging_reason", "lifetime_pd_ratio"} <= set(result.columns)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, Stage.STAGE_1), ("Stage 2", Stage.STAGE_2), ("stage_3", Stage.STAGE_3)],
)
def test_normalize_stage_accepts_common_labels(value: object, expected: Stage) -> None:
    assert normalize_stage(value) is expected

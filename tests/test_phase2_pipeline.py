import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.phase2 import (
    MODEL_FEATURES,
    deterministic_case_control_sample,
    prepare_adjacent_pairs,
    project_lifetime_curves,
    run_phase2,
    weighted_validation_metrics,
)


def test_adjacent_pairs_exclude_gaps_and_conflicting_loan_months() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["a", "a", "a", "a", "b", "b", "b"],
            "reporting_month": [
                202001,
                202001,
                202002,
                202004,
                202001,
                202001,
                202002,
            ],
            "state": [
                "CURRENT",
                "CURRENT",
                "30_DPD",
                "DEFAULTED",
                "CURRENT",
                "30_DPD",
                "DEFAULTED",
            ],
            "loan_age": [1, 1, 2, 4, 1, 1, 2],
        }
    )

    pairs = prepare_adjacent_pairs(panel)

    assert len(pairs) == 1
    assert pairs.loc[0, "loan_id"] == "a"
    assert pairs.loc[0, "from_state"] == "CURRENT"
    assert pairs.loc[0, "to_state"] == "30_DPD"
    assert pairs.loc[0, "target_month"] == 202002


def test_case_control_sampling_is_stable_and_preserves_non_current_rows() -> None:
    frame = pd.DataFrame(
        {
            "loan_id": [f"loan-{number:03d}" for number in range(100)]
            + ["event", "delinquent"],
            "current_month": [201101] * 102,
            "current_state": ["CURRENT"] * 101 + ["30_DPD"],
            "target_default": [0] * 100 + [1, 0],
        }
    )

    first = deterministic_case_control_sample(frame, modulus=5, seed=41)
    second = deterministic_case_control_sample(
        frame.sample(frac=1.0, random_state=7), modulus=5, seed=41
    )

    assert set(first["loan_id"]) == set(second["loan_id"])
    assert {"event", "delinquent"}.issubset(set(first["loan_id"]))
    sampled_current = first.loc[
        first["current_state"].eq("CURRENT")
        & first["target_default"].eq(0)
    ]
    assert 0 < len(sampled_current) < 100
    assert sampled_current["selection_probability"].eq(0.2).all()
    assert first.loc[first["loan_id"].eq("event"), "selection_probability"].item() == 1.0


def test_weighted_validation_metrics_reconcile_exact_values() -> None:
    metrics = weighted_validation_metrics(
        [0, 1],
        [0.2, 0.8],
        [1.0, 3.0],
    )

    assert metrics["weighted_event_rate"] == pytest.approx(0.75)
    assert metrics["weighted_mean_prediction"] == pytest.approx(0.65)
    assert metrics["weighted_brier_score"] == pytest.approx(0.04)
    assert metrics["weighted_roc_auc"] == pytest.approx(1.0)


def test_lifetime_projection_caps_term_and_forces_existing_default() -> None:
    class ConstantHazardModel:
        def predict_hazard_curve(self, features, horizon_months):
            return np.full((len(features), horizon_months), 0.1)

    features = pd.DataFrame(
        {
            column: [1.0, 2.0] if column in MODEL_FEATURES[:7] else ["A", "B"]
            for column in MODEL_FEATURES
        }
    )
    marginal, twelve_month, lifetime = project_lifetime_curves(
        ConstantHazardModel(),
        features,
        horizon_months=4,
        remaining_term_months=[2, 0],
        force_default=[False, True],
        batch_size=1,
    )

    np.testing.assert_allclose(marginal[0], [0.1, 0.09, 0.0, 0.0])
    np.testing.assert_allclose(marginal[1], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(twelve_month, [0.19, 1.0])
    np.testing.assert_allclose(lifetime, [0.19, 1.0])


def _synthetic_panel() -> pd.DataFrame:
    months = [201110, 201111, 201112, 201201, 201202, 201203]
    states = {
        "loan-0": ["CURRENT", "30_DPD", "DEFAULTED"],
        "loan-1": ["30_DPD", "30_DPD", "DEFAULTED"],
        "loan-2": ["CURRENT"] * 6,
        "loan-3": ["CURRENT", "CURRENT", "CURRENT", "30_DPD", "DEFAULTED"],
        "loan-4": ["CURRENT", "CURRENT", "CURRENT", "CURRENT", "30_DPD", "DEFAULTED"],
        "loan-5": ["30_DPD", "CURRENT", "CURRENT", "CURRENT", "CURRENT", "CURRENT"],
        "loan-6": ["CURRENT"] * 6,
        "loan-7": ["30_DPD", "CURRENT", "CURRENT", "CURRENT", "CURRENT", "CURRENT"],
    }
    rows = []
    for loan_number, (loan_id, loan_states) in enumerate(states.items()):
        for position, state in enumerate(loan_states):
            rows.append(
                {
                    "loan_id": loan_id,
                    "reporting_month": months[position],
                    "state": state,
                    "delinquency_status": "00",
                    "loan_age": str(60 + position),
                    "remaining_months_to_legal_maturity": str(300 - position),
                    "current_actual_upb": str(150000 - position * 500),
                    "current_interest_rate": "5.25",
                    "estimated_ltv": str(70 + loan_number),
                    "modification_flag": "N",
                    "orig_classic_fico": str(650 + 10 * loan_number),
                    "orig_original_ltv": str(70 + loan_number),
                    "orig_original_dti": "35",
                    "orig_original_interest_rate": "5.25",
                    "orig_original_loan_term": "360",
                    "orig_occupancy_status": "P",
                    "orig_loan_purpose": "P",
                }
            )
    return pd.DataFrame(rows)


def _synthetic_config(root: Path) -> ProjectConfig:
    return ProjectConfig(
        root=root,
        raw={
            "project": {"name": "phase2-test", "random_seed": 42},
            "paths": {
                "processed_data": "data/processed",
                "artifacts": "artifacts",
                "figures": "reports/figures",
            },
            "data": {"source_root": None},
            "phase0": {},
            "phase1": {
                "vintages": ["2005Q1"],
                "sample_seed": 41,
            },
            "modeling": {
                "development_end_month": 201112,
                "validation_end_month": 201203,
                "backtest_snapshot_month": 201203,
                "current_non_event_month_modulus": 1,
                "maximum_iterations": 1000,
            },
            "ifrs9": {"maximum_projection_months": 6},
        },
    )


def _write_phase1_checkpoint(config: ProjectConfig) -> None:
    panel = _synthetic_panel()
    processed = config.path("processed_data") / "phase1"
    artifacts = config.path("artifacts") / "phase1" / "quarters"
    processed.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    panel_path = processed / "panel_2005q1.parquet"
    panel.to_parquet(panel_path, index=False)
    summary = {
        "release": 47,
        "sample_seed": 41,
        "sampled_loans": panel["loan_id"].nunique(),
        "expansion_weight": 2.0,
        "panel": {"rows_written": len(panel)},
    }
    with (artifacts / "2005q1.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle)


def test_run_phase2_writes_restart_safe_model_and_curve_artifacts(tmp_path) -> None:
    config = _synthetic_config(tmp_path)
    _write_phase1_checkpoint(config)

    summary = run_phase2(config)

    assert summary["development"]["events"] >= 2
    assert summary["validation"]["events"] >= 2
    assert summary["model_fit"]["converged"]
    assert summary["snapshot_loans"] == 5
    assert all(Path(path).is_file() for path in summary["outputs"].values())
    development = pd.read_parquet(summary["outputs"]["development_hazard_sample"])
    validation = pd.read_parquet(summary["outputs"]["validation_hazard_rows"])
    assert development["target_month"].max() <= 201112
    assert validation["target_month"].min() > 201112

    current = pd.read_parquet(summary["outputs"]["current_pd_curves"])
    origination = pd.read_parquet(summary["outputs"]["origination_pd_curves"])
    snapshot = pd.read_parquet(summary["outputs"]["snapshot_loans"])
    assert current["loan_id"].is_unique
    assert origination["loan_id"].is_unique
    assert current["current_marginal_pd"].map(len).eq(6).all()
    assert origination["origination_marginal_pd"].map(len).eq(6).all()
    default_row = snapshot.loc[snapshot["is_default"]].iloc[0]
    assert default_row["current_lifetime_pd"] == pytest.approx(1.0)
    assert current["remaining_term_months"].equals(
        origination["remaining_term_months"]
    )

    model_path = Path(summary["outputs"]["hazard_model"])
    modified_time = model_path.stat().st_mtime_ns
    reused = run_phase2(config)
    assert reused == summary
    assert model_path.stat().st_mtime_ns == modified_time

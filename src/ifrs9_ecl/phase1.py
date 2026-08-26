"""Multi-vintage panel construction, EDA, and segmented roll rates."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ifrs9_ecl.archive import iter_origination_rows, iter_performance_rows
from ifrs9_ecl.config import ProjectConfig
from ifrs9_ecl.panel import write_monthly_panel
from ifrs9_ecl.reporting import save_transition_outputs
from ifrs9_ecl.sampling import HashSample, select_hash_sample
from ifrs9_ecl.schemas import ORIGINATION_COLUMNS, RELEASE_NUMBER
from ifrs9_ecl.states import DEFAULTED, DPD_90_PLUS, STATE_ORDER
from ifrs9_ecl.transitions import TransitionResult, estimate_transitions
from ifrs9_ecl.utils import utc_timestamp, write_json


def _write_origination_rows(sample: HashSample, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(column, pa.string()) for column in ORIGINATION_COLUMNS])
    rows = [row.as_dict() for row in sample.rows]
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    temporary.replace(path)


def _quarter_paths(config: ProjectConfig, vintage: str) -> dict[str, Path]:
    processed = config.path("processed_data") / "phase1"
    artifacts = config.path("artifacts") / "phase1" / "quarters"
    key = vintage.lower()
    return {
        "origination": processed / f"origination_{key}.parquet",
        "panel": processed / f"panel_{key}.parquet",
        "summary": artifacts / f"{key}.json",
    }


def _load_reusable_quarter(
    paths: dict[str, Path],
    *,
    vintage: str,
    maximum_loans: int,
    seed: int,
) -> dict[str, Any] | None:
    if not all(paths[name].is_file() for name in ("origination", "panel", "summary")):
        return None
    with paths["summary"].open(encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = {
        "release": RELEASE_NUMBER,
        "vintage": vintage,
        "requested_loans": maximum_loans,
        "sample_seed": seed,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None
    panel_rows = pq.ParquetFile(paths["panel"]).metadata.num_rows
    origination_rows = pq.ParquetFile(paths["origination"]).metadata.num_rows
    if panel_rows != summary["panel"]["rows_written"]:
        return None
    if origination_rows != summary["sampled_loans"]:
        return None
    return summary


def build_quarter_panel(
    config: ProjectConfig,
    vintage: str,
    *,
    maximum_loans: int,
    seed: int,
    batch_rows: int,
    reuse: bool,
) -> dict[str, Any]:
    """Select loans uniformly by stable hash and stream their complete histories."""
    paths = _quarter_paths(config, vintage)
    if reuse:
        existing = _load_reusable_quarter(
            paths,
            vintage=vintage,
            maximum_loans=maximum_loans,
            seed=seed,
        )
        if existing is not None:
            return existing

    archive_path = config.archive_path(vintage)
    sample = select_hash_sample(
        iter_origination_rows(archive_path),
        maximum_loans,
        seed=seed,
    )
    _write_origination_rows(sample, paths["origination"])
    lookup = {row["loan_id"]: row for row in sample.rows}
    stream = iter_performance_rows(
        archive_path,
        sample.loan_id_set,
        stop_after_selected=False,
        validate_loan_order=True,
    )
    with stream:
        panel = write_monthly_panel(
            stream,
            lookup,
            paths["panel"],
            vintage=vintage,
            batch_rows=batch_rows,
        )
    if not stream.diagnostics.completed_archive_scan:
        raise RuntimeError(f"{vintage} performance member was not scanned completely")
    if not stream.diagnostics.selection_complete:
        raise RuntimeError(f"{vintage} is missing one or more sampled loan histories")
    if panel.loan_ids != set(sample.loan_ids):
        missing = sorted(set(sample.loan_ids) - panel.loan_ids)[:10]
        raise RuntimeError(f"{vintage} sampled loans missing from panel: {missing}")

    panel_payload = panel.to_dict()
    panel_payload.pop("loan_ids")
    summary = {
        "release": RELEASE_NUMBER,
        "vintage": vintage,
        "archive": str(archive_path),
        "requested_loans": maximum_loans,
        "sampled_loans": len(sample.rows),
        "population_originations": sample.population_rows,
        "sampling_fraction": sample.sampling_fraction,
        "expansion_weight": sample.expansion_weight,
        "sample_seed": seed,
        "sample_method": "smallest stable 64-bit loan ID hashes",
        "performance_scan": asdict(stream.diagnostics),
        "panel": panel_payload,
        "outputs": {name: str(path) for name, path in paths.items() if name != "summary"},
    }
    write_json(summary, paths["summary"])
    return summary


def _numeric(values: pd.Series, unavailable: Iterable[float] = ()) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce").astype("float64")
    if unavailable:
        result = result.mask(result.isin(list(unavailable)))
    return result


def fico_band(values: pd.Series) -> pd.Series:
    numeric = _numeric(values, unavailable=(9999,))
    return pd.cut(
        numeric,
        bins=[-np.inf, 659, 699, 739, np.inf],
        labels=["Below 660", "660 to 699", "700 to 739", "740 plus"],
    ).astype("string").fillna("Unknown")


def ltv_band(values: pd.Series) -> pd.Series:
    numeric = _numeric(values, unavailable=(999,))
    return pd.cut(
        numeric,
        bins=[-np.inf, 60, 80, 90, np.inf],
        labels=["60 or below", "61 to 80", "81 to 90", "Above 90"],
    ).astype("string").fillna("Unknown")


def _adjacent_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    period = pd.to_numeric(work["reporting_month"], errors="raise").astype("int64")
    month = period % 100
    if not month.between(1, 12).all():
        raise ValueError("Panel contains an invalid reporting month")
    work["_ordinal"] = (period // 100) * 12 + month - 1
    work = work.sort_values(["loan_id", "_ordinal"], kind="stable")
    grouped = work.groupby("loan_id", sort=False, observed=True)
    work["from_state"] = grouped["state"].shift()
    work["_previous_ordinal"] = grouped["_ordinal"].shift()
    return work[
        work["from_state"].notna()
        & ((work["_ordinal"] - work["_previous_ordinal"]) == 1)
    ].rename(columns={"state": "to_state"})


def _segmented_transition_rows(
    pairs: pd.DataFrame,
    *,
    segment_column: str,
    segment_type: str,
    expansion_weight: float,
) -> pd.DataFrame:
    grouped = (
        pairs.groupby(
            [segment_column, "from_state", "to_state"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("sample_count")
        .reset_index()
        .rename(columns={segment_column: "segment"})
    )
    grouped.insert(0, "segment_type", segment_type)
    grouped["weighted_count"] = grouped["sample_count"] * expansion_weight
    return grouped


def _origination_feature_summary(frame: pd.DataFrame, vintage: str) -> pd.DataFrame:
    specifications = {
        "classic_fico": (9999,),
        "original_ltv": (999,),
        "original_cltv": (999,),
        "original_dti": (999,),
        "original_upb": (),
        "original_interest_rate": (),
    }
    rows = []
    for feature, unavailable in specifications.items():
        values = _numeric(frame[feature], unavailable)
        observed = values.dropna()
        rows.append(
            {
                "vintage": vintage,
                "feature": feature,
                "rows": len(values),
                "observed": len(observed),
                "missing_rate": float(values.isna().mean()),
                "mean": float(observed.mean()),
                "p25": float(observed.quantile(0.25)),
                "median": float(observed.median()),
                "p75": float(observed.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def analyze_quarter(
    config: ProjectConfig,
    quarter_summary: dict[str, Any],
) -> dict[str, Any]:
    """Create transition and EDA evidence for one completed quarter panel."""
    vintage = str(quarter_summary["vintage"])
    expansion_weight = float(quarter_summary["expansion_weight"])
    paths = _quarter_paths(config, vintage)
    columns = [
        "loan_id",
        "reporting_month",
        "state",
        "zero_balance_code",
        "actual_loss",
        "zero_balance_removal_upb",
        "orig_classic_fico",
        "orig_original_ltv",
    ]
    panel = pd.read_parquet(paths["panel"], columns=columns)
    transition_input = panel[["loan_id", "reporting_month", "state"]]
    transition = estimate_transitions(transition_input)
    pairs = _adjacent_pairs(
        panel[
            [
                "loan_id",
                "reporting_month",
                "state",
                "orig_classic_fico",
                "orig_original_ltv",
            ]
        ]
    )
    pairs["fico_band"] = fico_band(pairs["orig_classic_fico"])
    pairs["ltv_band"] = ltv_band(pairs["orig_original_ltv"])
    segmented = pd.concat(
        [
            _segmented_transition_rows(
                pairs,
                segment_column="fico_band",
                segment_type="Classic FICO",
                expansion_weight=expansion_weight,
            ),
            _segmented_transition_rows(
                pairs,
                segment_column="ltv_band",
                segment_type="Original LTV",
                expansion_weight=expansion_weight,
            ),
        ],
        ignore_index=True,
    )

    state_counts = panel["state"].value_counts().rename_axis("state").reset_index(name="rows")
    state_counts.insert(0, "vintage", vintage)
    state_counts["weighted_rows"] = state_counts["rows"] * expansion_weight
    monthly = (
        panel.groupby(["reporting_month", "state"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    monthly.insert(0, "vintage", vintage)
    monthly["weighted_rows"] = monthly["rows"] * expansion_weight

    loan_states = (
        panel.assign(value=True)
        .pivot_table(
            index="loan_id",
            columns="state",
            values="value",
            aggfunc="max",
            fill_value=False,
        )
        .astype(bool)
    )
    def state_flag(state: str) -> pd.Series:
        if state in loan_states:
            return loan_states[state]
        return pd.Series(False, index=loan_states.index)

    defaulted_loans = int(state_flag(DEFAULTED).sum())
    paid_off_loans = int(state_flag("PAID_OFF").sum())
    censored_loans = int(
        (state_flag("CENSORED_RPL") | state_flag("CENSORED_DEFECT")).sum()
    )
    actual_loss = _numeric(panel["actual_loss"])
    removal_upb = _numeric(panel["zero_balance_removal_upb"])
    loss_mask = actual_loss.notna()
    lgd_mask = loss_mask & removal_upb.gt(0)
    lgd = (actual_loss[lgd_mask] / removal_upb[lgd_mask]).rename("realized_lgd")
    lgd_rows = pd.DataFrame(
        {
            "vintage": vintage,
            "realized_lgd": lgd,
            "expansion_weight": expansion_weight,
        }
    )

    origination = pd.read_parquet(paths["origination"])
    feature_summary = _origination_feature_summary(origination, vintage)
    return {
        "vintage": vintage,
        "expansion_weight": expansion_weight,
        "transition": transition,
        "segmented": segmented,
        "state_counts": state_counts,
        "monthly": monthly,
        "feature_summary": feature_summary,
        "lgd_rows": lgd_rows,
        "loan_outcomes": {
            "sampled_loans": len(loan_states),
            "defaulted_loans": defaulted_loans,
            "paid_off_loans": paid_off_loans,
            "censored_loans": censored_loans,
            "weighted_defaulted_loans": defaulted_loans * expansion_weight,
            "weighted_paid_off_loans": paid_off_loans * expansion_weight,
            "weighted_censored_loans": censored_loans * expansion_weight,
            "actual_loss_rows": int(loss_mask.sum()),
            "actual_loss_sum": float(actual_loss[loss_mask].sum()),
            "weighted_actual_loss_sum": float(
                actual_loss[loss_mask].sum() * expansion_weight
            ),
            "realized_lgd_rows": int(lgd_mask.sum()),
            "realized_lgd_below_zero": int((lgd < 0).sum()),
            "realized_lgd_above_one": int((lgd > 1).sum()),
        },
    }


def _combine_segmented(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(
            ["segment_type", "segment", "from_state", "to_state"],
            observed=True,
        )[["sample_count", "weighted_count"]]
        .sum()
        .reset_index()
    )
    totals = grouped.groupby(
        ["segment_type", "segment", "from_state"], observed=True
    )["weighted_count"].transform("sum")
    grouped["probability"] = grouped["weighted_count"] / totals
    return grouped.sort_values(
        ["segment_type", "segment", "from_state", "to_state"]
    ).reset_index(drop=True)


def _make_phase1_figures(
    config: ProjectConfig,
    state_counts: pd.DataFrame,
    monthly: pd.DataFrame,
    lgd_rows: pd.DataFrame,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figures = config.path("figures")
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    mix = state_counts.pivot_table(
        index="vintage",
        columns="state",
        values="rows",
        aggfunc="sum",
        fill_value=0,
    )
    mix = mix.div(mix.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(12, 5))
    mix.plot(kind="bar", stacked=True, ax=axis, width=0.85)
    axis.set_ylabel("Share of monthly records")
    axis.set_xlabel("Origination quarter")
    axis.set_title("Monthly credit-state mix by origination quarter")
    axis.legend(title="State", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    path = figures / "phase1_state_mix_by_vintage.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs["state_mix_figure"] = str(path)

    time = monthly.pivot_table(
        index="reporting_month",
        columns="state",
        values="weighted_rows",
        aggfunc="sum",
        fill_value=0,
    )
    denominator = time.sum(axis=1).replace(0, np.nan)
    severe = (
        time.get(DPD_90_PLUS, pd.Series(0.0, index=time.index))
        + time.get(DEFAULTED, pd.Series(0.0, index=time.index))
    ) / denominator
    dates = pd.to_datetime(time.index.astype(str), format="%Y%m")
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(dates, severe * 100, color="#a61b1b", linewidth=1.5)
    axis.set_ylabel("90 plus or defaulted share, percent")
    axis.set_xlabel("Reporting month")
    axis.set_title("Severe delinquency across the sampled vintages")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = figures / "phase1_severe_delinquency_timeline.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    outputs["severe_delinquency_figure"] = str(path)

    if not lgd_rows.empty:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        clipped = lgd_rows["realized_lgd"].clip(-0.5, 2.0)
        axis.hist(clipped, bins=60, color="#355f8a", alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axvline(1, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Actual Loss divided by Zero Balance Removal UPB")
        axis.set_ylabel("Disposition records")
        axis.set_title("Raw realized severity, clipped only for display")
        axis.grid(alpha=0.2, axis="y")
        figure.tight_layout()
        path = figures / "phase1_realized_lgd_distribution.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs["realized_lgd_figure"] = str(path)
    return outputs


def run_phase1(config: ProjectConfig) -> dict[str, Any]:
    """Build every configured quarter and produce pooled Phase 1 evidence."""
    settings = config.raw["phase1"]
    vintages = [str(value).upper() for value in settings["vintages"]]
    maximum_loans = int(settings["loans_per_vintage"])
    seed = int(settings["sample_seed"])
    batch_rows = int(settings.get("parquet_batch_rows", 50_000))
    reuse = bool(settings.get("reuse_completed_quarters", True))
    config.ensure_output_directories()

    quarter_summaries = []
    for vintage in vintages:
        print(f"Phase 1 panel: {vintage}", flush=True)
        quarter_summary = build_quarter_panel(
            config,
            vintage,
            maximum_loans=maximum_loans,
            seed=seed,
            batch_rows=batch_rows,
            reuse=reuse,
        )
        quarter_summaries.append(quarter_summary)
        print(
            f"Completed {vintage}: {quarter_summary['sampled_loans']} loans, "
            f"{quarter_summary['panel']['rows_written']} monthly rows",
            flush=True,
        )

    analyses = []
    for summary in quarter_summaries:
        print(f"Phase 1 analysis: {summary['vintage']}", flush=True)
        analyses.append(analyze_quarter(config, summary))
    table_directory = config.path("artifacts") / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    pooled_counts = pd.DataFrame(0.0, index=STATE_ORDER, columns=STATE_ORDER)
    transition_rows = []
    for analysis in analyses:
        result: TransitionResult = analysis["transition"]
        weight = float(analysis["expansion_weight"])
        pooled_counts = pooled_counts.add(result.counts * weight, fill_value=0.0)
        long = result.long.copy()
        long.insert(0, "vintage", analysis["vintage"])
        transition_rows.append(long)
    row_totals = pooled_counts.sum(axis=1).replace(0, np.nan)
    pooled_probabilities = pooled_counts.div(row_totals, axis=0)
    pooled_counts.index.name = pooled_probabilities.index.name = "from_state"
    pooled_counts.columns.name = pooled_probabilities.columns.name = "to_state"
    pooled_diagnostics = {
        key: int(sum(a["transition"].diagnostics.to_dict()[key] for a in analyses))
        for key in analyses[0]["transition"].diagnostics.to_dict()
        if key != "largest_gap_months"
    }
    pooled_diagnostics["largest_gap_months"] = max(
        a["transition"].diagnostics.largest_gap_months for a in analyses
    )
    pooled_outputs = save_transition_outputs(
        pooled_counts,
        pooled_probabilities,
        pooled_diagnostics,
        table_directory=table_directory,
        figure_directory=config.path("figures"),
        prefix="phase1_pooled_weighted",
    )

    vintage_transitions = pd.concat(transition_rows, ignore_index=True)
    vintage_transitions.to_csv(
        table_directory / "phase1_vintage_transitions.csv", index=False
    )
    segmented = _combine_segmented([a["segmented"] for a in analyses])
    segmented.to_csv(
        table_directory / "phase1_segmented_transitions.csv", index=False
    )
    state_counts = pd.concat([a["state_counts"] for a in analyses], ignore_index=True)
    state_counts.to_csv(table_directory / "phase1_state_counts.csv", index=False)
    monthly = pd.concat([a["monthly"] for a in analyses], ignore_index=True)
    monthly = (
        monthly.groupby(["reporting_month", "state"], observed=True)[
            ["rows", "weighted_rows"]
        ]
        .sum()
        .reset_index()
    )
    monthly.to_csv(table_directory / "phase1_monthly_state_counts.csv", index=False)
    feature_summary = pd.concat(
        [a["feature_summary"] for a in analyses], ignore_index=True
    )
    feature_summary.to_csv(
        table_directory / "phase1_origination_feature_summary.csv", index=False
    )
    lgd_rows = pd.concat([a["lgd_rows"] for a in analyses], ignore_index=True)
    lgd_rows.to_parquet(
        config.path("processed_data") / "phase1" / "realized_lgd_sample.parquet",
        index=False,
    )
    figure_outputs = _make_phase1_figures(config, state_counts, monthly, lgd_rows)

    outcome_rows = [a["loan_outcomes"] for a in analyses]
    sampled_loans = sum(row["sampled_loans"] for row in outcome_rows)
    summary = {
        "run_completed_at_utc": utc_timestamp(),
        "release": RELEASE_NUMBER,
        "vintages": vintages,
        "sample_method": "deterministic smallest stable hash within every quarter",
        "sampled_loans": sampled_loans,
        "population_originations": sum(
            item["population_originations"] for item in quarter_summaries
        ),
        "performance_rows": sum(item["panel"]["rows_written"] for item in quarter_summaries),
        "actual_loss_rows_reconciled": sum(
            item["panel"]["actual_loss_reconciled_rows"] for item in quarter_summaries
        ),
        "actual_loss_rows_outside_tolerance": sum(
            item["panel"]["actual_loss_outside_tolerance_rows"]
            for item in quarter_summaries
        ),
        "sampled_defaulted_loans": sum(row["defaulted_loans"] for row in outcome_rows),
        "weighted_defaulted_loans": sum(
            row["weighted_defaulted_loans"] for row in outcome_rows
        ),
        "sampled_paid_off_loans": sum(row["paid_off_loans"] for row in outcome_rows),
        "sampled_censored_loans": sum(row["censored_loans"] for row in outcome_rows),
        "sample_actual_loss_sum": sum(row["actual_loss_sum"] for row in outcome_rows),
        "weighted_actual_loss_sum": sum(
            row["weighted_actual_loss_sum"] for row in outcome_rows
        ),
        "realized_lgd": {
            "rows": len(lgd_rows),
            "mean": float(lgd_rows["realized_lgd"].mean()),
            "median": float(lgd_rows["realized_lgd"].median()),
            "p25": float(lgd_rows["realized_lgd"].quantile(0.25)),
            "p75": float(lgd_rows["realized_lgd"].quantile(0.75)),
            "below_zero": int((lgd_rows["realized_lgd"] < 0).sum()),
            "above_one": int((lgd_rows["realized_lgd"] > 1).sum()),
        },
        "transitions": pooled_diagnostics,
        "key_weighted_transition_probabilities": {
            "current_to_30_dpd": float(
                pooled_probabilities.loc["CURRENT", "30_DPD"]
            ),
            "30_dpd_to_current": float(
                pooled_probabilities.loc["30_DPD", "CURRENT"]
            ),
            "30_dpd_to_60_dpd": float(
                pooled_probabilities.loc["30_DPD", "60_DPD"]
            ),
            "60_dpd_to_90_plus": float(
                pooled_probabilities.loc["60_DPD", "90_PLUS"]
            ),
            "90_plus_to_defaulted": float(
                pooled_probabilities.loc["90_PLUS", "DEFAULTED"]
            ),
        },
        "quarter_summaries": quarter_summaries,
        "outputs": {
            **pooled_outputs,
            **figure_outputs,
            "vintage_transitions": str(
                table_directory / "phase1_vintage_transitions.csv"
            ),
            "segmented_transitions": str(
                table_directory / "phase1_segmented_transitions.csv"
            ),
            "monthly_state_counts": str(
                table_directory / "phase1_monthly_state_counts.csv"
            ),
            "origination_feature_summary": str(
                table_directory / "phase1_origination_feature_summary.csv"
            ),
        },
    }
    summary_path = config.path("artifacts") / "phase1_summary.json"
    write_json(summary, summary_path)
    return summary

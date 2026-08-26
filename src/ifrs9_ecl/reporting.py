"""Phase output tables and figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ifrs9_ecl.config import PROJECT_ROOT
from ifrs9_ecl.utils import write_json

_matplotlib_cache = PROJECT_ROOT / "artifacts" / ".matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_transition_outputs(
    counts: pd.DataFrame,
    probabilities: pd.DataFrame,
    diagnostics: dict[str, Any],
    *,
    table_directory: Path,
    figure_directory: Path,
    prefix: str,
) -> dict[str, str]:
    """Persist the empirical transition evidence in machine and human readable forms."""
    table_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    counts_path = table_directory / f"{prefix}_transition_counts.csv"
    probabilities_path = table_directory / f"{prefix}_transition_probabilities.csv"
    diagnostics_path = table_directory / f"{prefix}_transition_diagnostics.json"
    figure_path = figure_directory / f"{prefix}_transition_matrix.png"
    counts.to_csv(counts_path)
    probabilities.to_csv(probabilities_path, float_format="%.10f")
    write_json(diagnostics, diagnostics_path)

    outgoing = counts.sum(axis=1) > 0
    incoming = counts.sum(axis=0) > 0
    plot_probabilities = probabilities.loc[outgoing, incoming]
    width = max(8.0, 1.05 * len(plot_probabilities.columns) + 3.0)
    height = max(5.0, 0.8 * len(plot_probabilities.index) + 2.0)
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        plot_probabilities,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Monthly transition probability"},
        ax=axis,
    )
    axis.set_xlabel("Next monthly state")
    axis.set_ylabel("Current monthly state")
    axis.set_title("Empirical one-month delinquency transitions")
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "counts": str(counts_path),
        "probabilities": str(probabilities_path),
        "diagnostics": str(diagnostics_path),
        "figure": str(figure_path),
    }

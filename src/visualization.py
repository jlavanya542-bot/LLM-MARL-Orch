from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def _bar(summary: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    data = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(data["scheduler"], data[metric])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(results: pd.DataFrame, output_dir: str | Path) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main = results[~results["scheduler"].str.contains("::")].copy()
    summary = main.groupby("scheduler", as_index=False).mean(numeric_only=True)

    paths = {}
    specs = [
        ("sla_violation_rate_pct", "SLA violations (%)",
         "SLA Violation Rate Across Scheduling Models", "figure_sla_violation.png"),
        ("energy_wh_per_container", "Energy per container (Wh)",
         "Energy Consumption Across Scheduling Models", "figure_energy.png"),
        ("fairness_index", "Jain fairness index",
         "Workload Fairness Across Scheduling Models", "figure_fairness.png"),
        ("load_imbalance_score", "Load imbalance score",
         "Load Imbalance Across Scheduling Models", "figure_load_imbalance.png"),
        ("scheduling_latency_ms", "Online decision latency (ms)",
         "Scheduling Decision Latency", "figure_scheduling_latency.png"),
    ]
    for metric, ylabel, title, filename in specs:
        path = output / filename
        _bar(summary, metric, ylabel, title, path)
        paths[metric] = str(path)
    return paths

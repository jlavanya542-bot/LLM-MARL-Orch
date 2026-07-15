from __future__ import annotations

import pandas as pd


def summarize_ablations(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate full-model and component-removal experiments."""
    mask = results["scheduler"].str.startswith("LLM-MARL-Orch")
    subset = results.loc[mask].copy()
    metrics = [
        "sla_violation_rate_pct",
        "energy_wh_per_container",
        "fairness_index",
        "load_imbalance_score",
        "scheduling_latency_ms",
        "mean_reward",
    ]
    summary = subset.groupby("scheduler")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        col if isinstance(col, str) else "_".join([x for x in col if x])
        for col in summary.columns
    ]
    return summary

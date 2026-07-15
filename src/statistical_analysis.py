from __future__ import annotations

from math import sqrt
from typing import Iterable, List

import numpy as np
import pandas as pd


def _paired_ttest(a: np.ndarray, b: np.ndarray):
    try:
        from scipy.stats import ttest_rel
        stat, p = ttest_rel(a, b, nan_policy="omit")
        return float(stat), float(p)
    except Exception:
        diff = a - b
        if len(diff) < 2 or np.std(diff, ddof=1) == 0:
            return 0.0, 1.0
        stat = float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff))))
        # Normal approximation fallback.
        p = float(np.exp(-0.717 * abs(stat) - 0.416 * stat * stat))
        return stat, min(max(p, 0.0), 1.0)


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def aggregate_results(results: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    metrics = [
        "sla_violation_rate_pct",
        "energy_wh_per_container",
        "carbon_g_per_container",
        "mean_response_time_ms",
        "scheduling_latency_ms",
        "semantic_refresh_latency_ms",
        "fairness_index",
        "load_imbalance_score",
        "mean_reward",
    ]
    rows = []
    z = 1.96 if abs(confidence_level - 0.95) < 1e-6 else 1.96
    for scheduler, group in results.groupby("scheduler"):
        row = {"scheduler": scheduler, "runs": int(len(group))}
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = z * std / sqrt(max(len(values), 1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(results: pd.DataFrame, proposed: str = "LLM-MARL-Orch",
                 baseline: str = "DeepSched") -> pd.DataFrame:
    metrics = [
        "sla_violation_rate_pct",
        "energy_wh_per_container",
        "carbon_g_per_container",
        "mean_response_time_ms",
        "fairness_index",
        "load_imbalance_score",
        "scheduling_latency_ms",
    ]
    left = results[results["scheduler"] == proposed].set_index("seed")
    right = results[results["scheduler"] == baseline].set_index("seed")
    common = sorted(set(left.index).intersection(right.index))
    rows = []
    for metric in metrics:
        a = left.loc[common, metric].astype(float).to_numpy()
        b = right.loc[common, metric].astype(float).to_numpy()
        stat, p = _paired_ttest(a, b)
        baseline_mean = float(np.mean(b)) if len(b) else float("nan")
        proposed_mean = float(np.mean(a)) if len(a) else float("nan")
        relative = (
            100.0 * (baseline_mean - proposed_mean) / abs(baseline_mean)
            if baseline_mean not in (0.0, -0.0) else float("nan")
        )
        rows.append({
            "metric": metric,
            "proposed": proposed,
            "baseline": baseline,
            "paired_runs": len(common),
            "proposed_mean": proposed_mean,
            "baseline_mean": baseline_mean,
            "absolute_difference": proposed_mean - baseline_mean,
            "relative_improvement_pct": relative,
            "t_statistic": stat,
            "p_value": p,
            "cohens_d_paired": _cohens_d_paired(a, b),
        })
    return pd.DataFrame(rows)

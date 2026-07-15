from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ablation import summarize_ablations
from src.config import load_config
from src.evaluator import run_benchmark
from src.reproducibility import set_global_seed, write_manifest
from src.statistical_analysis import aggregate_results, paired_tests
from src.visualization import generate_figures
from src.workload_generator import save_public_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the reproducible LLM-MARL-Orch simulation."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Run a small smoke experiment.")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate the public dataset without benchmarking.")
    parser.add_argument("--skip-ablations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    output = root / cfg["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    master_seed = int(cfg["reproducibility"]["master_seed"])
    set_global_seed(master_seed, bool(cfg["reproducibility"]["deterministic"]))

    print("[1/5] Generating versioned public dataset...")
    save_public_dataset(cfg, output, master_seed)
    if args.generate_only:
        write_manifest(root, output / "artifact_manifest.sha256")
        print(f"Dataset generated in: {output}")
        return

    if args.quick:
        repetitions = int(cfg["experiment"]["quick_repetitions"])
        workload_count = int(cfg["experiment"]["quick_workloads"])
    else:
        repetitions = int(cfg["experiment"]["full_repetitions"])
        workload_count = int(cfg["dataset"]["workload_count"])

    seeds = list(cfg["reproducibility"]["experiment_seeds"])[:repetitions]

    print(f"[2/5] Running {repetitions} repetitions with {workload_count} workloads...")
    results = run_benchmark(
        cfg, seeds, workload_count, output,
        include_ablations=not args.skip_ablations
    )
    results.to_csv(output / "result_summary.csv", index=False)

    print("[3/5] Computing uncertainty estimates and paired statistical tests...")
    aggregate = aggregate_results(
        results, float(cfg["outputs"]["confidence_level"])
    )
    aggregate.to_csv(output / "aggregate_results.csv", index=False)
    tests = paired_tests(results)
    tests.to_csv(output / "statistical_tests.csv", index=False)
    ablations = summarize_ablations(results)
    ablations.to_csv(output / "ablation_results.csv", index=False)

    print("[4/5] Generating publication-ready figures...")
    if bool(cfg["outputs"]["generate_figures"]):
        generate_figures(results, output)

    manuscript_values = {
        "status": "reported_in_manuscript_not_hard_coded_as_reproduced_results",
        "reported_values": {
            "sla_violation_rate_pct": 3.2,
            "energy_wh_per_container": 0.84,
            "fairness_index": 0.94,
            "online_scheduling_latency_ms": 6.7,
        },
        "interpretation_note": (
            "Reproduced values depend on the declared seed, workload generator, "
            "model configuration, and hardware. The evaluation scripts compute all "
            "relative improvements directly and do not force these target values."
        ),
    }
    (output / "expected_results.json").write_text(
        json.dumps(manuscript_values, indent=2), encoding="utf-8"
    )

    print("[5/5] Writing integrity manifest...")
    write_manifest(root, output / "artifact_manifest.sha256")
    print(f"Completed. Results are available in: {output}")


if __name__ == "__main__":
    main()

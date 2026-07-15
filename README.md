# LLM-MARL-Orch

LLM-MARL-Orch is a reproducible, simulation-based implementation of semantic-aware multi-agent container scheduling for heterogeneous cloud-edge systems. The framework combines a cached semantic objective controller with decentralized cooperative reinforcement-learning agents to optimize service-level compliance, energy consumption, carbon impact, load balance, and workload fairness.

The repository accompanies the study titled **Large Language Model Guided Multiagent Scheduling for Energy Efficient Cloud Native Systems**. The implementation is deliberately explicit about its scope: the reported evaluation is conducted in a controlled Kubernetes-inspired simulator and is not presented as a production Kubernetes deployment.

## Main capabilities

The code generates a versioned public dataset containing 100 heterogeneous nodes, 5,000 timestamped workloads, and 10,000 semantic prompt-response samples. It provides chronological train, validation, and test partitions without temporal overlap. Six schedulers are evaluated under identical workload traces and random seeds: Kubernetes-style least-loaded placement, an energy-aware heuristic, a single-agent learning scheduler, cooperative MARL, an LLM-guided heuristic, and the complete LLM-MARL-Orch model.

The evaluation reports SLA violation rate, energy per container, carbon impact, response time, online scheduling latency, semantic refresh latency, cache-hit rate, fairness, load imbalance, feasibility, drift events, and reward. Ablation studies isolate semantic guidance, MARL, fairness, drift detection, and static objective weights. Statistical analysis includes mean, standard deviation, 95% confidence intervals, paired tests, relative improvement, and paired effect size.

## Repository structure

```text
.
├── src
│   ├── workload_generator.py
│   ├── cloud_edge_environment.py
│   ├── semantic_agent.py
│   ├── marl_agent.py
│   ├── reward_model.py
│   ├── schedulers.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── ablation.py
│   ├── statistical_analysis.py
│   └── visualization.py
├── artifacts
├── config.yaml
├── run_reproducibility.py
├── validate_repository.py
├── DATASET_CARD.md
├── REPRODUCIBILITY.md
├── MANUSCRIPT_REVISION_GUIDE.md
├── CODE_AVAILABILITY.md
├── DATA_AVAILABILITY.md
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository validation

```bash
python validate_repository.py
```

The validation script checks required files, configuration integrity, dataset generation, workload-class validity, and accidental secret exposure.

## Generate the public dataset

```bash
python run_reproducibility.py --generate-only
```

This creates:

```text
artifacts/nodes.csv
artifacts/workloads.csv
artifacts/semantic_prompt_corpus.csv
artifacts/dataset_metadata.json
artifacts/feature_schema.json
```

## Quick reproducibility run

```bash
python run_reproducibility.py --quick
```

The quick mode uses two declared seeds and 600 workloads to verify the end-to-end pipeline.

## Full reproducibility run

```bash
python run_reproducibility.py
```

The full configuration uses 5,000 workloads and ten independent seeds. Results are written to `artifacts/result_summary.csv`, `artifacts/aggregate_results.csv`, `artifacts/statistical_tests.csv`, and `artifacts/ablation_results.csv`. Publication-ready figures are generated from the computed CSV files.

## Semantic inference modes

The default backend is `deterministic_cached`. It maps structured workload semantics to normalized objective weights and stores them in `artifacts/semantic_weight_cache.json`. This mode is intended for exact, offline reproducibility and does not require an API key.

The optional `local_transformer` backend attempts to use a model already available in the local Hugging Face cache. It does not silently download a model. When the requested local model is unavailable, the code falls back to the deterministic semantic controller.

Online scheduling latency and semantic refresh latency are measured separately. This distinction is necessary because semantic outputs can be generated asynchronously and reused by the low-latency scheduling loop.

## Experimental fairness

Every scheduler receives the same node topology, workload trace, temporal split, and seed. Reported improvements are calculated programmatically from experimental outputs. The code does not force results to match values reported in the manuscript.

## Security

No credentials are required. Do not commit passwords, API keys, private datasets, or institutional secrets. The `.gitignore` file excludes common credential and environment files.



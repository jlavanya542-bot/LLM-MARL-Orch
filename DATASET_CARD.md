# Dataset Card

## Dataset name

LLM-MARL-Orch Synthetic Cloud-Edge Scheduling Dataset

## Purpose

The dataset supports reproducible evaluation of container-placement policies under heterogeneous cloud-edge resources, dynamic arrivals, workload bursts, energy scarcity, and semantic scheduling constraints.

## Files

`nodes.csv` contains the simulated infrastructure. It records cluster location, node type, compute and memory capacities, network capacity, idle and peak power, carbon intensity, renewable-energy ratio, and reliability.

`workloads.csv` contains timestamped container requests. Each record includes workload class, resource demands, duration, latency budget, energy sensitivity, edge-locality preference, priority, stress regime, and a structured semantic description.

`semantic_prompt_corpus.csv` contains 10,000 prompt-target examples. Target values are normalized objective weights for SLA compliance, energy, load imbalance, fairness, and carbon impact.

## Workload classes

Latency-sensitive workloads represent real-time analytics and edge-local services. Throughput-centric workloads represent batch and ETL processing. Transactional workloads represent low-jitter service endpoints. Hybrid workloads represent changing microservice chains with mixed objectives.

## Generation

The dataset is generated deterministically by `src/workload_generator.py` using the seed and parameters declared in `config.yaml`. The complete dataset can be regenerated with:

```bash
python run_reproducibility.py --generate-only
```

## Splitting protocol

The data are sorted by simulated arrival time and separated chronologically into training, validation, and test partitions. No record appears in more than one partition. The default proportions are 70%, 10%, and 20%.

## Privacy and ethics

The dataset is fully synthetic. It contains no personal information, user identifiers, communication content, protected health information, or confidential operational logs.

## Limitations

The data approximate cloud-native scheduling behavior but do not reproduce every property of Google Borg, Alibaba Cluster Trace, or a production Kubernetes cluster. Conclusions must therefore be described as controlled simulation results.

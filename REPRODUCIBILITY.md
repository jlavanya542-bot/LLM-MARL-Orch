# Reproducibility Protocol

Use Python 3.10 or later and install the packages listed in `requirements.txt`. Run `python validate_repository.py` before beginning an experiment.

The default experiment is configured in `config.yaml`. Ten fixed seeds are declared to support paired comparisons across schedulers. Each seed generates one node topology and one timestamped workload trace. All schedulers evaluated under a given seed receive the same topology and trace.

The workload sequence is split chronologically. Adaptive schedulers learn on the training partition and are measured on the held-out test partition. Validation records are reserved for future hyperparameter selection and are not merged into the test set.

Execute the full study with:

```bash
python run_reproducibility.py
```

Execute a small end-to-end verification with:

```bash
python run_reproducibility.py --quick
```

The main experiment writes seed-level measurements to `artifacts/result_summary.csv`. Aggregate means, standard deviations, and 95% confidence intervals are stored in `artifacts/aggregate_results.csv`. Paired significance tests against DeepSched are stored in `artifacts/statistical_tests.csv`. Component-removal experiments are stored in `artifacts/ablation_results.csv`.

Semantic refresh latency is distinct from online scheduling latency. The first measures the cost of producing a new semantic objective vector. The second measures the placement decision made with an already available or cached vector. A manuscript should not combine these quantities.

The file `artifacts/artifact_manifest.sha256` contains cryptographic hashes for source files, configuration files, dataset files, result tables, and documentation. Regenerating results may update the manifest because result files and semantic cache contents change.

Hardware, operating system, Python version, and package versions should be reported when final results are produced. Small numerical differences may arise from processor architecture and linear-algebra implementation even when seeds are fixed.

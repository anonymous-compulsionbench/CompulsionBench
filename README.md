# CompulsionBench: Auditing Compulsion-Risk Proxies in Sequential Recommendation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-brightgreen.svg)]()

This repository contains the official codebase and results bundle for **CompulsionBench: A Benchmark for Auditing Compulsion-Risk Proxies in Sequential Recommendation** (NeurIPS 2026, Datasets and Benchmarks Track).

## Overview

CompulsionBench is a reproducible benchmark designed to ask when engagement-maximizing recommendation policies interact with long-horizon user dynamics to increase observable proxies such as prolonged sessions, elevated night-time use, or over-cap exposure. 

The repository provides two plug-compatible user-response backends:
1. **CompulsionBench-Param**: A transparent parametric simulator used as the reference environment, modeling latent habit, fatigue, and self-control.
2. **CompulsionBench-LLM**: An LLM-augmented semantic robustness backend that perturbs the item-response kernel using language models (e.g., Qwen 2.5) via bounded semantic anchoring.

Both backends share the same observation space, action space, calibration targets, and standardized scorecard, separating observable compulsion-risk proxies from simulator-internal diagnostics.

## Repository Structure

- `compulsionbench/`: The core Python library, RL environment (Gymnasium), agent implementations (PPO, Lagrangian PPO, baselines), and execution scripts.
- `scripts/`: Submission scripts and data aggregation bash scripts used to produce the final 4-seed cluster results.
- `results/`: The official evaluated results bundle, containing scorecards, calibration plots, convergence logs, and mechanism ablations presented in the manuscript.
- `config_calibrated.json`: The frozen, Optuna-calibrated hyperparameters used across all experiments.

## Installation & Usage

For full instructions on how to set up the Python environment, download the KuaiRand and KuaiRec datasets, run the public-log calibration pipeline, and reproduce the paper results, please see the detailed technical guide:

👉 **[Read the CompulsionBench Technical Guide](compulsionbench/README.md)**

## Citation

If you use CompulsionBench in your research, please cite our paper:

```bibtex
@inproceedings{compulsionbench2026,
  title={CompulsionBench: A Benchmark for Auditing Compulsion-Risk Proxies in Sequential Recommendation},
  author={Anonymous},
  booktitle={Thirty-ninth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Misuse Note

This code is intended for auditing and benchmarking recommendation policies under explicit assumptions. It should not be used as deployment guidance for maximizing harmful engagement.

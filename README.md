# FTSE MIB Option Pricing

**Heston Monte Carlo, baseline neural networks and differential learning under sparse market data.**

This research project reconstructs and tests an options-pricing thesis workflow. It combines audited quote preparation, Heston calibration and simulation, and a neural price function whose Greeks are obtained through automatic differentiation.

**Relationship to the thesis:** this is a revised implementation of a master's thesis project, not an archive of the code or results originally submitted. The reconstruction includes subsequent corrections and new development experiments, assisted by AI coding tools. Reported results belong to this revised pilot; see [Provenance](docs/PROVENANCE.md) for the distinction.

The public code includes a **synthetic demo and data-free tests**. The empirical pilot uses licensed OptionMetrics IvyDB Europe data accessed through WRDS; those data, trained market-data checkpoints and the private thesis are excluded.

## What the project demonstrates

- Consistent price, strike, rate and Greek units, with explicit missing-label handling.
- Separation of calibration and development quotes by strike group.
- Analytic Heston references and full-truncation Monte Carlo with Sobol sampling and antithetic paths.
- Matched price-only and differential neural networks with one price output and derivative-coupled losses.
- Checks for numerical consistency, seed variation and economic violations, including unfavorable results.

Here **BNN means baseline neural network**, not Bayesian neural network.

## Run without market data

Tested with Python 3.12 on macOS arm64. Use a separate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/demo.py --output output/demo.json
```

The demo checks a synthetic Heston price and an untrained network's price derivatives. It does **not** reproduce the empirical table below. See [Reproduction](docs/REPRODUCTION.md) for the distinction between the public demo and the licensed-data research workflow.

## Development results

The bounded pilot uses **664 training quotes and 156 development quotes** across ten dates. Both neural models use identical data, initialization and minibatch ordering within each of three seeds. The development quotes also select the neural stopping epoch.

| Model | Price MAE | Price RMSE |
|---|---:|---:|
| Heston Monte Carlo, 20,000 paths | **39.35** | **61.00** |
| Baseline network, mean of three runs | 127.98 | 163.52 |
| Differential network, mean of three runs | 160.11 | 198.97 |

Errors are in index points. Neural values average run-level metrics; they are not ensemble scores. The MC row uses the first preselected scramble, not the best of several runs. Its numerical checks use separately reported independent repetitions.

DML lowers Delta/Gamma/Theta proxy-label MAE in all three runs, but has higher price MAE in all three and greater variation in price predictions. Both neural models produce some negative prices and other economic violations. These findings do **not** establish a general advantage for DML.

See [Results and limitations](docs/RESULTS.md) and [Methodology](docs/METHODOLOGY.md).

## Repository guide

| Location | Purpose |
|---|---|
| `scripts/demo.py` | Self-contained synthetic example. |
| `scripts/heston_benchmark.py`, `heston_sobol.py` | Analytic reference, IID MC and scrambled Sobol/antithetic MC. |
| `scripts/neural_pricing.py` | Scalar price network, automatic Greeks and masked differential losses. |
| Other `scripts/` | Quote checks, calibration and research-stage orchestration; private inputs required. |
| `tests/` | Synthetic tests of pricing, cleaning, group separation, derivatives and training controls. |
| `config/` | Documented empirical pilot settings. |
| `docs/` | Method, results, reproduction and code provenance. |

## Current scope

Completed: bounded three-model development comparison, three-seed neural check, numerical MC validation and derivative consistency checks.

Still open: final chronological evaluation, controlled data-sparsity experiments, the Vega-coordinate mismatch and any hedging claims. The public release is a documented research pilot, not a completed empirical validation or a production pricing system.

No project-wide redistribution license has been selected. Third-party dependencies retain their own licenses; see [Code and data provenance](docs/PROVENANCE.md).

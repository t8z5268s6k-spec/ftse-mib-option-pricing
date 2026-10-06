# Reproduction guide

## Public, data-free workflow

Create a Python 3.12 environment and install `requirements.txt`, then run:

```bash
python -m unittest discover -s tests -v
python scripts/demo.py --output output/demo.json
```

The tests use synthetic fixtures. The demo uses an illustrative Heston scenario and a randomly initialized price network; the network is not presented as a trained pricer. Four independent scrambles with 8,192 paths each provide a small, balanced-base Sobol example. This differs deliberately from the empirical 20,000-path specification and does not produce the empirical results.

Generated files go into ignored local directories. The demo refuses to overwrite an existing output; choose another filename to repeat it. The tested environment is macOS arm64, Python 3.12.14 and CPU float64. Other platforms have not been tested for this release. Exact dependency versions are recorded in the requirements files.

## Licensed-data research workflow

The remaining scripts expose the research implementation, but their `main` entry points expect verified local artifacts. They are **not a one-command public empirical reproduction**. Neither public test success nor the synthetic demo reproduces the reported FTSE MIB table.

Re-running that table requires authorized OptionMetrics IvyDB Europe data, the same quote definitions, EUR rate/dividend inputs, the frozen membership, and locally regenerated integrity records. Providing similarly named files or disabling hash checks is not equivalent to reproduction.

| Stage | Entry points | Required private inputs |
|---|---|---|
| Audit and clean | `audit_raw_data.py`, `prepare_audited_data.py` | Original export, recorded schema and source verification. |
| Tick/carry checks | `audit_tick_training.py`, `validate_tick_inputs.py`, `check_forward_consistency.py` | Training tick quotes, daily anchors, rate/dividend histories and prior audit artifacts. |
| Heston calibration | `prepare_benchmark_protocol_v2.py`, `run_heston_v2.py`, `verify_heston_v2.py` | Approved quote diagnostics, membership, protocol and verification records. |
| Neural preparation | `prepare_neural_pilot.py` | Verified Heston pilot rows, fitted parameters and raw vendor labels. |
| Neural comparison | `run_neural_pilot.py`, `check_neural_seed_stability.py` | Prepared rows, fit-only scalers and prerequisite verification records. |
| MC comparison | `compare_mc_neural_pilot.py` | Verified Heston controls, neural results and prior MC checks. |

The scripts use project-relative paths under `data/`, `results/current/` and `local_metadata/`. These directories are excluded from publication. Earlier manual acquisition steps and the complete private verification history are not included. Read each entry point's prerequisites before running it; many stages deliberately refuse to overwrite results.

The public functional APIs can be used with independently constructed inputs without reproducing the original vendor pipeline:

- `heston_benchmark.analytic_price`: European Heston with explicit date, rate and dividend yield.
- `heston_sobol.sobol_replicate` and `summarize_replicates`: repeated scrambled Monte Carlo and between-scramble uncertainty.
- `neural_pricing.PriceNetwork`, `price_and_greeks`, `pilot_loss`: a single price function and masked derivative training losses.
- `prepare_neural_pilot.bsm_price_greeks`, `label_audit`, `fit_scalers`: unit reconstruction, label eligibility and fit-only normalization.

## Data and evaluation boundaries

Do not commit vendor downloads, quote-level exports, trained market-data weights, private documents, credentials or local provenance. The repository's ignore rules are a guardrail; review the staged file list before publishing.

Do not interpret the current development scores as chronological holdout results. Do not substitute vendor IV for V0, enable a vendor-Vega loss against a different volatility coordinate, or relabel the eight-scramble MC mean as a 20,000-path result.

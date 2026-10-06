# Development results and limitations

Results below summarize the verified local research runs dated October 5, 2026. Licensed quote-level inputs, predictions and market-trained weights are not distributed. These tables cannot be regenerated from the synthetic demo.

## Neural seed check

All runs use the same 664 fit and 156 development quotes. Price errors are index points.

| Seed | BNN MAE | DML MAE | BNN RMSE | DML RMSE |
|---|---:|---:|---:|---:|
| 42 | 133.42 | 170.13 | 166.46 | 208.71 |
| 43 | 121.18 | 125.53 | 157.18 | 153.55 |
| 44 | 129.33 | 184.67 | 166.94 | 234.64 |
| Mean | 127.98 | 160.11 | 163.52 | 198.97 |
| Sample standard deviation | 6.23 | 30.82 | 5.50 | 41.41 |

These are means and standard deviations of run-level metrics, not ensemble scores or confidence intervals. Mean contract-level price standard deviation across seeds is 37.74 for BNN and 77.20 for DML. Initialization and data order both change with the seed.

On the same 93 eligible development rows, mean Greek proxy MAE across seeds is:

| Label | BNN | DML |
|---|---:|---:|
| Delta | 0.10409 | 0.05272 |
| Gamma | 0.00004120 | 0.00002558 |
| Theta, points/year | 440.85 | 250.00 |

DML improves each of these proxy errors in all three runs. It has worse price MAE in all three; its RMSE is better only at seed 43. BNN generates 5–6 negative development prices per run, and DML 3–5. Both retain price-bound and Greek-shape violations. Predictions are not clipped.

## Actual Monte Carlo comparison

| Heston estimate, same 156 rows | MAE | RMSE | Inside bid/ask |
|---|---:|---:|---:|
| First preselected 20,000-path scramble | 39.35 | 61.00 | 103/156 |
| Mean of eight 20,000-path scrambles | 38.96 | 60.76 | 105/156 |
| Analytic reference | 38.90 | 60.47 | 109/156 |

Across eight complete 20,000-path panels, MAE ranges from 37.21 to 40.29 and RMSE from 58.47 to 62.27. All 156 mean-price checks pass the diagnostic tolerance of four scramble-mean standard errors plus 0.03 at normalized spot 100. Maximum mean-price discrepancy from analytic Heston is still 10.06 index points, so this is not a one-point precision guarantee. No rare-payoff warnings occur in this sample.

## Interpretation

- Development rows also select neural stopping epochs. They are not untouched test data.
- Heston is fitted separately per date, while the networks learn across dates and receive the calibrated V0 input. The learning problems are not identical.
- The date sample was used in earlier data diagnostics. WRDS carry is option-derived.
- Greek labels are vendor model proxies on a non-random subset of quotes. Better agreement does not establish hedging performance.
- The seed-42 baseline reached the 500-epoch cap. Optimization convergence is not established for all runs, and three seeds do not establish statistical significance.
- Vega, final chronological evaluation and controlled data-sparsity tests remain open.

The pilot supports a transparent empirical finding: this DML setup better matches the Greek proxies but does not improve price MAE over the baseline. It does not support a universal ranking of the methods.

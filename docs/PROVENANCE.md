# Code and data provenance

This release contains the reconstructed and tested research implementation from the thesis working project. Historical notebook exports, original five-output neural scripts, reference notebooks, personal paths, private documents and licensed quote-level datasets are excluded.

## Relationship to the original thesis

This repository is a revised implementation of a master's thesis project, not an exact archive of the original submission. Subsequent work corrected unit handling and data preparation, introduced per-date Heston calibration and checked Monte Carlo simulation, and replaced separately predicted Greeks with derivatives of one neural price function. The development results reported here were obtained in the reconstruction and must not be attributed to the original thesis submission.

AI coding tools assisted with the reconstruction, code changes, checks and documentation. The reported numerical checks support the bounded pilot; they do not establish final out-of-sample performance. Additional training experiments, a complete synthetic differential-learning benchmark and the final chronological evaluation remain future work rather than completed improvements.

## Publication contents

Most Python modules and all selected tests are copied unchanged from the validated working implementation. `calibration_helpers.py` extracts only the shared pricing, metrics and digest functions from an earlier calibration script. Two Heston modules redirect their helper imports to that file. The older script's executable entry point is not included. These publication edits do not alter pricing or optimization formulas. `demo.py` is a new, entirely synthetic demonstration.

Aggregate result tables summarize local verified runs; they are not raw market data or evidence of an independently reproduced public dataset. The publication intentionally does not redistribute vendor manuals, contract-level observations, model checkpoints fitted to those observations, or the private thesis.

## Method references and dependencies

- [Huge and Savine, Differential Machine Learning](https://arxiv.org/abs/2005.02347): learning from function values and differential labels. This project implements its own empirical price/Greek workflow; it does not redistribute the authors' notebooks.
- [QuantLib](https://www.quantlib.org/): analytic Heston pricing reference. The tests include an attributed stress-case value from the QuantLib test suite.
- [PyTorch automatic differentiation](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html): derivative computation through the scalar network output.
- [SciPy Sobol documentation](https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.stats.qmc.Sobol.html): randomized quasi-Monte Carlo and power-of-two balance considerations.

Third-party packages are installed as dependencies; their source code is not vendored here. Each retains its own license. **No project-wide redistribution license has been selected.** Public visibility should not be interpreted as a blanket grant covering dependency code or proprietary data.

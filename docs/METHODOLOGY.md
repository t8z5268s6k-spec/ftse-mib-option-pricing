# Methodology

## Empirical task and data

The task is conditional same-day pricing of European FTSE MIB calls and puts. The shared price target is the bid/ask midpoint in index points. Strike values are converted from the vendor's thousandths convention; maturity is ACT/365F. Rates and dividend yields are continuous decimal inputs, without curve extrapolation.

The available historical source spans 2019–2023. The current pilot uses ten dates from the training period, not a full-period experiment. The recorded bid and ask times must match a fixed daily anchor chosen by quote counts, with the latest time resolving ties. Timestamp equality does not establish independently synchronized feeds. Call/put members of a date/expiry/strike group stay in the same partition; every fifth distinct strike is a development group.

Activity flags are retained for descriptive analysis, not used to remove difficult quotes. The current low-activity indicator is `(Volume < 10) AND (OpenInterest < 50)`.

## Heston benchmark

One five-parameter Heston fit per date uses training quotes only and fixed WRDS carry. Three predefined optimization starts minimize `(model price - midpoint) / max(halfspread, 1 point)`. The numerical reference is QuantLib analytic Heston. One stress-date fit reaches the upper volatility-of-variance bound; start agreement does not prove parameter identification.

Monte Carlo uses full truncation for the auxiliary variance and a log-Euler stock update. Both updates use start-of-step variance. Call and put payoffs are computed directly from the same simulated terminal stock prices.

The thesis's 64 steps/year did not pass the bounded numerical check; 256 steps/year was adopted provisionally and checked on all 156 development prices. Each empirical scramble uses 10,000 Sobol points plus 10,000 antithetic paths. The non-power-of-two base count loses the usual Sobol balance guarantee. Eight independent scrambles support uncertainty estimates; standard errors are computed between scrambles, not across dependent paths.

The headline MC result uses the preselected first 20,000-path scramble. The mean of eight scrambles uses 160,000 paths per price and is reported separately. Strikes sharing the same term inputs share random numbers. Numerical uncertainty does not include parameter, input or discretization uncertainty.

## Matched neural models

Both models take `S, K, T, V0, R, Q, IsCall`. V0 is the date's calibrated Heston start variance. The option's own implied volatility, bid, ask, spread and Greek labels are not input features. WRDS carry is itself option-derived, so this is not an independent forecasting setup.

The networks share hidden widths 128/64/32, Softplus activations, Xavier initialization, fixed fit-only standardization and one scalar price output. Adam uses learning rate 0.001, batches of 512, a 500-epoch cap and patience 25. There is no batch normalization or dropout. Both models select checkpoints using development price MSE. Each paired run uses the same seed for weights and minibatch permutations.

The baseline loss uses price only. DML additionally matches automatic Delta, Gamma and calendar-time Theta to eligible vendor BSM proxy labels. A label requires compatible price/spot bases and numerical reconstruction. There are 369 eligible training rows and 93 eligible development rows; all 664/156 price rows remain in the price loss.

The raw conventions are Delta = ∂P/∂S, Gamma = ∂²P/∂S² and Theta = −∂P/∂T per year. With input/output standard deviations sS, sT and sP, standardized derivative targets are Delta·sS/sP, Gamma·sS²/sP and −Theta·sT/sP. Derivative losses are further scaled by training-label RMS. Active weights are price 1, Delta 0.5, Gamma 0.1 and Theta 0.2. Missing labels are masked before residual arithmetic.

Vendor Vega is ∂P_BSM/∂σ_IV, per unit decimal implied volatility. It is not generally `2 sqrt(V0) × ∂P_network/∂V0`; no vendor Vega loss is enabled. The latter derivative is exposed under the explicit name `InitialVolSensitivity`.

## Relation to the thesis

The original research question is retained. The working thesis contains conflicting market-versus-Heston targets and a five-output implementation that does not enforce price/Greek consistency. This reconstruction uses a common market target and derivatives of a single price function. Softplus permits a meaningful Gamma; the prior ReLU/affine-input design does not provide a useful second derivative almost everywhere. These changes and the omission of an invalid Vega target must be reflected in any final thesis revision.

This follows the differential-label principle described by [Huge and Savine, Differential Machine Learning](https://arxiv.org/abs/2005.02347). Mathematical derivative consistency does not guarantee correct market sensitivities, positive prices or arbitrage freedom.

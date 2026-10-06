"""European Heston pricing in index points, with explicit continuous r and q.

Analytic reference: QuantLib. Independent MC: log-stock/full-truncation Euler.
Parameters and flat curves are scenarios unless explicitly calibrated elsewhere.
"""
from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import QuantLib as ql
from scipy.special import ndtr


@dataclass(frozen=True)
class HestonParameters:
    v0: float = .04
    kappa: float = 1.5
    theta: float = .04
    sigma: float = .3
    rho: float = -.7

    def validate(self):
        if not all(math.isfinite(x) for x in (self.v0, self.kappa, self.theta, self.sigma, self.rho)):
            raise ValueError('Heston parameters must be finite.')
        if self.v0 < 0 or self.kappa <= 0 or self.theta <= 0 or self.sigma < 0 or abs(self.rho) >= 1:
            raise ValueError('Require v0>=0, kappa/theta>0, sigma>=0 and abs(rho)<1.')


def validate_inputs(spot, strike, t, r, q, params):
    params.validate()
    if not all(math.isfinite(x) for x in (spot, strike, t, r, q)) or spot <= 0 or strike <= 0 or t < 0:
        raise ValueError('Require finite inputs, positive spot/strike and nonnegative maturity.')


def deterministic_variance_price(spot, strike, t, r, q, params, kind):
    """Exact sigma=0 limit, including mean-reverting deterministic variance."""
    integrated = params.theta*t + (params.v0-params.theta)*(-math.expm1(-params.kappa*t))/params.kappa
    s, k = spot*math.exp(-q*t), strike*math.exp(-r*t)
    if integrated <= 0:
        return max(s-k, 0) if kind == 'C' else max(k-s, 0)
    root = math.sqrt(integrated)
    d1 = (math.log(s/k) + integrated/2)/root
    if kind == 'C':
        return float(s*ndtr(d1)-k*ndtr(d1-root))
    return float(k*ndtr(root-d1)-s*ndtr(-d1))


def analytic_price(spot, strike, days, r, q, params=HestonParameters(), kind='C',
                   valuation_date=date(2020, 1, 1), integration_order=192):
    """ACT/365F maturity and fixed-date flat curves; no date inferred from today."""
    if not isinstance(days, (int, np.integer)) or days < 0 or kind not in ('C', 'P'):
        raise ValueError('days must be a nonnegative integer and kind must be C or P.')
    t = days/365
    validate_inputs(spot, strike, t, r, q, params)
    if days == 0:
        return max(spot-strike, 0) if kind == 'C' else max(strike-spot, 0)
    if params.sigma == 0:
        return deterministic_variance_price(spot, strike, t, r, q, params, kind)
    expiry = valuation_date + timedelta(days=int(days))
    today = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    expiry_ql = ql.Date(expiry.day, expiry.month, expiry.year)
    previous_date = ql.Settings.instance().evaluationDate
    try:
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        risk_free = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
        dividend = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
        process = ql.HestonProcess(risk_free, dividend, ql.QuoteHandle(ql.SimpleQuote(spot)),
                                   params.v0, params.kappa, params.theta, params.sigma, params.rho)
        model = ql.HestonModel(process)
        # Fixed Gauss-Laguerre quadrature also handles the small-sigma test where
        # tight adaptive quadrature exhausts iterations from cancellation noise.
        engine = ql.AnalyticHestonEngine(model, integration_order)
        option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if kind == 'C' else ql.Option.Put, strike),
                                  ql.EuropeanExercise(expiry_ql))
        option.setPricingEngine(engine)
        result = float(option.NPV())
    finally:
        ql.Settings.instance().evaluationDate = previous_date
    if not math.isfinite(result):
        raise ArithmeticError('Nonfinite analytic Heston price.')
    return result  # Do not conceal quadrature failures by clipping negative prices.


def monte_carlo(spot, strike, days, r, q, params=HestonParameters(), *, paths=100000, steps=256, seed=42):
    """IID paths; SE and intervals measure sampling error, not time-step bias.

    Both stock and variance use variance at the start of the step. The auxiliary
    variance can be negative; its positive part enters drift and diffusion.
    Calls and puts are estimated independently from the same terminal prices.
    """
    if not isinstance(days, (int, np.integer)) or days < 0:
        raise ValueError('days must be a nonnegative integer.')
    if not isinstance(paths, int) or paths < 2 or not isinstance(steps, int) or steps < 1:
        raise ValueError('Require integer paths>=2 and steps>=1.')
    t = days/365
    validate_inputs(spot, strike, t, r, q, params)
    rng = np.random.default_rng(seed)
    log_s = np.full(paths, math.log(spot), dtype=float)
    variance = np.full(paths, params.v0, dtype=float)
    dt = t/steps
    for _ in range(steps):
        z_s, z_i = rng.standard_normal((2, paths))
        v_plus = np.maximum(variance, 0)
        diffusion = np.sqrt(v_plus*dt)
        log_s += (r-q-.5*v_plus)*dt + diffusion*z_s
        variance += params.kappa*(params.theta-v_plus)*dt + params.sigma*diffusion*(params.rho*z_s + math.sqrt(1-params.rho**2)*z_i)
    terminal = np.exp(log_s)
    disc = math.exp(-r*t)
    def estimate(values):
        mean = float(values.mean())
        se = float(values.std(ddof=1)/math.sqrt(paths))
        nonzero = int(np.count_nonzero(values))
        # A rare-event payoff with few/no hits does not justify a normal interval.
        # Thirty is a warning threshold, not a proof of asymptotic normality.
        sparse = nonzero < 30
        return {'price': mean, 'se': se, 'ci95_low': None if sparse else mean-1.96*se,
                'ci95_high': None if sparse else mean+1.96*se,
                'nonzero_paths': nonzero, 'sparse_payoff_warning': sparse}
    call = estimate(disc*np.maximum(terminal-strike, 0))
    put = estimate(disc*np.maximum(strike-terminal, 0))
    discounted_spot = estimate(disc*terminal)
    return {'call': call, 'put': put, 'discounted_spot': discounted_spot,
            'discounted_spot_target': spot*math.exp(-q*t), 'paths': paths, 'steps': steps, 'seed': seed,
            'parity_residual': call['price']-put['price']-(spot*math.exp(-q*t)-strike*disc)}

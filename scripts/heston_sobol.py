"""Scrambled Sobol/antithetic full-truncation log-Euler Heston diagnostic.

Uncertainty is estimated across independent scrambles, never across dependent
paths. Non-power-of-two base samples are explicitly marked as unbalanced.
"""
import math
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t
from heston_benchmark import HestonParameters, validate_inputs


def sobol_replicate(spot, strike, days, r, q, params=HestonParameters(), *,
                    paths=20000, steps_per_year=64, seed=42):
    if not isinstance(days, (int, np.integer)) or days < 0:
        raise ValueError('days must be a nonnegative integer')
    if not isinstance(paths, int) or paths < 4 or paths % 2:
        raise ValueError('paths must be an even integer >=4')
    if not isinstance(steps_per_year, int) or steps_per_year < 1:
        raise ValueError('steps_per_year must be a positive integer')
    time = days / 365
    validate_inputs(spot, strike, time, r, q, params)
    steps = max(1, math.ceil(time * steps_per_year))
    if 2 * steps > 21201:
        raise ValueError('Sobol dimension exceeds supported maximum')
    base = paths // 2
    balanced = base & (base - 1) == 0
    # Prefix preserves the thesis path count, but loses Sobol balance guarantees.
    uniforms = qmc.Sobol(2 * steps, scramble=True, seed=seed).random_base2(
        (base - 1).bit_length())[:base]
    normals = ndtri(np.clip(uniforms, np.finfo(float).eps, 1-np.finfo(float).eps))
    del uniforms
    log_s = np.full(paths, math.log(spot))
    variance = np.full(paths, params.v0, dtype=float)
    dt = time / steps
    negative = 0
    for j in range(steps):
        zs = np.concatenate((normals[:, 2*j], -normals[:, 2*j]))
        zi = np.concatenate((normals[:, 2*j+1], -normals[:, 2*j+1]))
        negative += int(np.count_nonzero(variance < 0))
        vp = np.maximum(variance, 0)
        diffusion = np.sqrt(vp * dt)
        log_s += (r-q-.5*vp)*dt + diffusion*zs
        variance += params.kappa*(params.theta-vp)*dt + params.sigma*diffusion*(
            params.rho*zs + math.sqrt(1-params.rho**2)*zi)
    terminal = np.exp(log_s)
    discount = math.exp(-r*time)
    call = discount*np.maximum(terminal-strike, 0)
    put = discount*np.maximum(strike-terminal, 0)
    return dict(call=float(call.mean()), put=float(put.mean()),
                discounted_spot=float(discount*terminal.mean()),
                nonzero_call=int(np.count_nonzero(call)), nonzero_put=int(np.count_nonzero(put)),
                negative_variance_fraction=negative/(paths*steps), steps=steps,
                paths=paths, seed=seed, balanced_base_sample=balanced)


def summarize_replicates(replicates):
    if len(replicates) < 2 or len({x['seed'] for x in replicates}) != len(replicates):
        raise ValueError('Require at least two independent scramble seeds')
    if len({(x['steps'], x['paths']) for x in replicates}) != 1:
        raise ValueError('Replicates must share time and path settings')
    result = {}
    for key in ('call', 'put', 'discounted_spot'):
        values = np.array([x[key] for x in replicates])
        mean, se = float(values.mean()), float(values.std(ddof=1)/math.sqrt(len(values)))
        sparse = key != 'discounted_spot' and any(x['nonzero_'+key] < 30 for x in replicates)
        width = float(student_t.ppf(.975, len(values)-1)*se)
        result[key] = dict(price=mean, se=se, sparse_payoff_warning=sparse,
                           approximate_ci95_low=None if sparse else mean-width,
                           approximate_ci95_high=None if sparse else mean+width)
    return result

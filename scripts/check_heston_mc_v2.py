"""Bounded numerical audit; never refits or consumes chronological holdouts."""
from pathlib import Path
from datetime import date
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
from heston_benchmark import HestonParameters, analytic_price
from heston_sobol import sobol_replicate, summarize_replicates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/current/heston_pilot_v2'
OUT = ROOT/'results/current/heston_mc_validation_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    protected = {str(p.relative_to(ROOT)): digest(p) for p in SOURCE.rglob('*') if p.is_file()}
    protected.update(json.loads((SOURCE/'protected_before.json').read_text()))
    # Saved calibration starts: no fresh optimization or data-dependent bounds.
    stability = []
    names = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    ranges = np.array([1-.0001, 15-.05, 1-.0001, 3-.01, 1.96])
    for file in sorted(SOURCE.glob('*/summary.json')):
        s = json.loads(file.read_text())
        vectors = np.array([[a['parameters'][k] for k in names] for a in s['attempts']])
        quotes = pd.read_csv(file.parent/'predictions.csv')
        fit = quotes[quotes.Role == 'fit']
        start_prices = []
        for attempt in s['attempts']:
            params = HestonParameters(**attempt['parameters'])
            start_prices.append([analytic_price(x.UnderlyingMid, x.StrikePoints, int(x.Days),
                x.r, x.q, params, x.CallPut, date.fromisoformat(s['date'])) for x in fit.itertuples()])
        stability.append(dict(date=s['date'], all_starts_success=all(a['success'] for a in s['attempts']),
            cost_range=float(np.ptp([a['cost'] for a in s['attempts']])),
            parameter_ranges=dict(zip(names, np.ptp(vectors, axis=0).tolist())),
            max_parameter_range_fraction=float((np.ptp(vectors, axis=0)/ranges).max()),
            max_fit_price_range_points=float(np.ptp(start_prices, axis=0).max()),
            feller_margin=s['feller_margin'], near_bounds=s['near_bounds']))
    cases = []
    for day in ['2019-05-02', '2020-05-04', '2021-12-17']:
        s = json.loads((SOURCE/day/'summary.json').read_text())
        f = pd.read_csv(SOURCE/day/'predictions.csv')
        f = f[f.Role == 'fit'].copy()
        for days in [int(f.Days.min()), int(f.Days.max())]:
            group = f[f.Days == days].copy()
            group['distance'] = abs(np.log(group.StrikePoints/(group.UnderlyingMid*np.exp((group.r-group.q)*days/365))))
            row = group.sort_values(['distance', 'StrikePoints', 'ExportRow']).iloc[0]
            cases.append(dict(date=day, days=days, source_export_row=int(row.ExportRow),
                original_spot=float(row.UnderlyingMid), original_strike=float(row.StrikePoints),
                spot=100., strike=float(100*row.StrikePoints/row.UnderlyingMid),
                r=float(row.r), q=float(row.q), parameters=s['parameters']))
    protocol = dict(cases=cases, paths_per_replicate=20000, independent_scrambles=8,
        steps_per_year=[64, 256, 1024], base_seed=58200,
        method='Full-truncation log-Euler; scrambled Sobol plus antithetic normals',
        sample_selection='First pilot date, sigma-boundary stress date, first new-rate-regime pilot date; shortest and longest maturity; fit strike nearest WRDS forward',
        balanced_sobol_base=False, warning='10000 base points are a truncated power-of-two net; no balance guarantee',
        diagnostic_tolerance='absolute MC-analytic error <= 4 scramble-mean SE + 0.03, normalized spot=100; not proof of convergence',
        uncertainty='Across 8 scrambles: 160000 total paths per contract/time setting. Approximate Student-t interval excludes discretization bias.',
        protected_sha256=protected)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (OUT/'start_stability.json').write_text(json.dumps(stability, indent=2)+'\n')
    rows, all_replicates = [], []
    started = time.monotonic()
    for case_index, case in enumerate(cases):
        params = HestonParameters(**case['parameters'])
        inputs = [case[k] for k in ('spot', 'strike', 'days', 'r', 'q')]
        targets = {kind: analytic_price(*inputs, params, code, date.fromisoformat(case['date']))
                   for kind, code in [('call', 'C'), ('put', 'P')]}
        targets['discounted_spot'] = 100*math.exp(-case['q']*case['days']/365)
        for level, steps in enumerate(protocol['steps_per_year']):
            replicates = [sobol_replicate(*inputs, params, steps_per_year=steps,
                seed=58200+case_index*1000+level*100+rep) for rep in range(8)]
            all_replicates.append(dict(case_index=case_index, steps_per_year=steps, replicates=replicates))
            estimates = summarize_replicates(replicates)
            for kind, estimate in estimates.items():
                error = estimate['price']-targets[kind]
                rows.append(dict(case_index=case_index, date=case['date'], days=case['days'],
                    steps_per_year=steps, actual_steps=replicates[0]['steps'], kind=kind,
                    analytic=targets[kind], **estimate, error=error,
                    error_original_points=error*case['original_spot']/100,
                    tolerance=4*estimate['se']+.03,
                    diagnostic_pass=abs(error)<=4*estimate['se']+.03 and not estimate['sparse_payoff_warning'],
                    negative_variance_fraction=float(np.mean([x['negative_variance_fraction'] for x in replicates]))))
            pd.DataFrame(rows).to_csv(OUT/'comparison.csv', index=False)
            (OUT/'replicates.json').write_text(json.dumps(all_replicates, indent=2)+'\n')
            print(f"Completed {case['date']} {case['days']}d at {steps}/year", flush=True)
    changed = [p for p, sha in protected.items() if digest(ROOT/p) != sha]
    if changed:
        raise RuntimeError(f'Protected files changed: {changed}')
    result = pd.DataFrame(rows)
    summary = dict(protocol_sha256=digest(OUT/'protocol.json'),
        protected_files_unchanged=len(protected), elapsed_seconds=time.monotonic()-started,
        by_steps_per_year={str(steps): dict(option_checks=int(len(g[g.kind!='discounted_spot'])),
            option_pass=int(g[g.kind!='discounted_spot'].diagnostic_pass.sum()),
            martingale_pass=int(g[g.kind=='discounted_spot'].diagnostic_pass.sum()),
            max_abs_option_error_normalized=float(g[g.kind!='discounted_spot'].error.abs().max()))
            for steps, g in result.groupby('steps_per_year')},
        script_sha256={p.name:digest(p) for p in [Path(__file__), ROOT/'scripts/heston_sobol.py']})
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

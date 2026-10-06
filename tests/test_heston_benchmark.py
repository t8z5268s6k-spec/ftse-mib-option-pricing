import sys
import unittest
from pathlib import Path
from datetime import date
from dataclasses import replace
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from heston_benchmark import HestonParameters, analytic_price, monte_carlo
import QuantLib as ql


class HestonTests(unittest.TestCase):
    def test_expiry_payoffs(self):
        self.assertEqual(analytic_price(100, 90, 0, .01, .03), 10)
        self.assertEqual(analytic_price(100, 110, 0, .01, .03, kind='P'), 10)

    def test_known_black_scholes_limit(self):
        # S=K=100, r=5%, q=0, vol=20%, T=1: standard BSM reference.
        p = HestonParameters(sigma=0)
        self.assertAlmostEqual(analytic_price(100, 100, 365, .05, 0, p), 10.450583572185565, places=10)
        near = replace(p, sigma=1e-4, rho=0)
        self.assertAlmostEqual(analytic_price(100, 100, 365, .05, 0, near), 10.450583572185565, places=5)

    def test_external_heston_stress_reference(self):
        # QuantLib test-suite Kahl/Jaeckel example; ten ACT/365F years.
        p = HestonParameters(v0=.16, theta=.16, kappa=1, sigma=2, rho=-.8)
        self.assertAlmostEqual(analytic_price(100, 200, 3650, 0, 0, p), 4.95212, delta=1e-4)

    def test_parity_and_discounted_bounds_grid(self):
        for r, q in [(.01, 0), (-.01, .03)]:
            for days in [1, 30, 365, 1825]:
                for strike in [60,100,140]:
                    c = analytic_price(100, strike, days, r, q)
                    p = analytic_price(100, strike, days, r, q, kind='P')
                    s, k = 100*math.exp(-q*days/365), strike*math.exp(-r*days/365)
                    self.assertAlmostEqual(c-p, s-k, delta=1e-7)
                    self.assertGreaterEqual(c, max(s-k,0)-1e-7)
                    self.assertLessEqual(c, s+1e-7)
                    self.assertGreaterEqual(p, max(k-s,0)-1e-7)
                    self.assertLessEqual(p, k+1e-7)

    def test_dividend_direction_and_homogeneity(self):
        c = analytic_price(100,100,365,.01,0)
        self.assertLess(analytic_price(100,100,365,.01,.03), c)
        self.assertAlmostEqual(analytic_price(25000,25000,365,.01,0), 250*c, places=7)

    def test_valuation_date_is_restored(self):
        old = ql.Settings.instance().evaluationDate
        analytic_price(100,100,365,.01,.02,valuation_date=date(2019,4,2))
        self.assertEqual(ql.Settings.instance().evaluationDate, old)

    def test_bad_inputs_raise(self):
        for kw in [dict(spot=-1), dict(days=-1), dict(kind='X'), dict(r=float('nan'))]:
            args = dict(spot=100,strike=100,days=365,r=.01,q=0)
            args.update(kw)
            with self.assertRaises(ValueError):
                analytic_price(**args)

    def test_mc_reproducibility_and_discounted_stock(self):
        args = dict(spot=100,strike=100,days=365,r=.01,q=.03,paths=40000,steps=256,seed=3101)
        a, b = monte_carlo(**args), monte_carlo(**args)
        self.assertEqual(a, b)
        self.assertLess(abs(a['discounted_spot']['price']-a['discounted_spot_target']), 4*a['discounted_spot']['se'])
        ref = analytic_price(100,100,365,.01,.03)
        self.assertLess(abs(a['call']['price']-ref), 4*a['call']['se']+.03)

    def test_zero_sample_payoffs_do_not_claim_exact_zero_value(self):
        estimate = monte_carlo(100,100000,1,.01,0,paths=1000,steps=8,seed=77)['call']
        self.assertEqual(estimate['nonzero_paths'],0)
        self.assertTrue(estimate['sparse_payoff_warning'])
        self.assertIsNone(estimate['ci95_high'])


if __name__ == '__main__':
    unittest.main()

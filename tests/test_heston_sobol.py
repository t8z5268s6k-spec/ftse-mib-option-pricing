import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from heston_benchmark import HestonParameters, analytic_price
from heston_sobol import sobol_replicate, summarize_replicates


class SobolTests(unittest.TestCase):
    def test_constant_variance_and_reproducibility(self):
        p = HestonParameters(v0=.04, theta=.04, sigma=0)
        runs = [sobol_replicate(100, 100, 365, .03, .01, p,
                paths=8192, steps_per_year=1, seed=s) for s in range(8)]
        self.assertEqual(runs[0], sobol_replicate(100,100,365,.03,.01,p,paths=8192,steps_per_year=1,seed=0))
        estimate = summarize_replicates(runs)
        for key, kind in [('call','C'),('put','P')]:
            self.assertLess(abs(estimate[key]['price']-analytic_price(100,100,365,.03,.01,p,kind)), .02)
        self.assertTrue(runs[0]['balanced_base_sample'])
        # Same terminal paths must satisfy sample parity, not imposed analytic parity.
        self.assertAlmostEqual(runs[0]['call']-runs[0]['put'],runs[0]['discounted_spot']-100*np.exp(-.03))

    def test_zero_volatility_and_expiry(self):
        p = HestonParameters(v0=0, theta=.04, sigma=0)
        x = sobol_replicate(100,95,0,.03,.01,p,paths=20)
        self.assertAlmostEqual(x['call'],5)
        self.assertEqual(x['put'],0)
        self.assertFalse(x['balanced_base_sample'])

    def test_validation_and_scramble_uncertainty(self):
        with self.assertRaises(ValueError): sobol_replicate(100,100,30,0,0,paths=21)
        with self.assertRaises(ValueError): sobol_replicate(100,100,-1,0,0)
        with self.assertRaises(ValueError): sobol_replicate(100,100,30,0,0,steps_per_year=0)
        a = sobol_replicate(100,100,30,0,0,paths=128,seed=1)
        b = sobol_replicate(100,100,30,0,0,paths=128,seed=2)
        with self.assertRaises(ValueError): summarize_replicates([a,a])
        s = summarize_replicates([a,b])
        self.assertAlmostEqual(s['call']['se'],abs(a['call']-b['call'])/2)


if __name__ == '__main__': unittest.main()

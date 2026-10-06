import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from check_forward_consistency import intersect_forward


class ForwardConsistencyTests(unittest.TestCase):
    def test_common_forward_can_fix_vendor_outside(self):
        r,_,_=intersect_forward([100,110],[-2,-11],[3,-5],1,106)
        self.assertEqual(r["CommonForwardLow"],99)
        self.assertEqual(r["CommonForwardHigh"],103)
        self.assertTrue(r["ForwardFeasible"])
        self.assertFalse(r["VendorWithinAll"])
        self.assertAlmostEqual(r["ClosestCompatibleForward"],103.000001)
        self.assertEqual(r["VendorOutsidePairs"],2)

    def test_disjoint_intervals_cannot_be_fixed_by_one_forward(self):
        r,_,_=intersect_forward([100,100],[0,4],[2,6],1,103)
        self.assertFalse(r["ForwardFeasible"])
        self.assertEqual(r["MinimaxParityDistance"],1)
        self.assertIsNone(r["ClosestCompatibleForward"])
        self.assertNotEqual(r["BindingLowIndex"],r["BindingHighIndex"])

    def test_discount_conversion_and_binding_boundary(self):
        r,lo,hi=intersect_forward([100,110],[-1,-5],[1,-3],.5,100)
        np.testing.assert_allclose(lo,[98,100]);np.testing.assert_allclose(hi,[102,104])
        self.assertTrue(r["VendorWithinAll"])
        self.assertEqual(r["CommonForwardLow"],100)
        self.assertEqual(r["CommonForwardHigh"],102)

    def test_nonpositive_forward_is_not_feasible(self):
        r,_,_=intersect_forward([100],[-110],[-105],1,100)
        self.assertFalse(r["ForwardFeasible"])
        self.assertEqual(r["MinimaxParityDistance"],5)

    def test_invalid_or_empty_inputs_rejected(self):
        for args in [([],[],[],1,100),([100],[2],[1],1,100),([100],[0],[1],0,100)]:
            with self.assertRaises(ValueError):intersect_forward(*args)


if __name__=="__main__":unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from validate_tick_inputs import bsm, timestamp_flags, choose_anchors, within_anchor, pair_panel


class TickInputTests(unittest.TestCase):
    def test_missing_midnight_and_out_of_day_are_not_synchronized(self):
        d=timestamp_flags(pd.DataFrame({"BidTime":["0:00:00",None,"24:00:00","17:30:00"],
                                        "AskTime":["0:00:00","17:30:00","24:00:00","17:30:00"]}))
        self.assertEqual(d.ValidTimes.tolist(),[False,False,False,True])

    def test_anchor_uses_count_then_latest_not_price(self):
        d=pd.DataFrame({"Date":["2020-01-02"]*4,"BidTime":["17:30:00"]*2+["17:31:00"]*2,
                        "AskTime":["17:30:00"]*2+["17:31:00"]*2})
        a=choose_anchors(timestamp_flags(d))
        self.assertEqual(a.AnchorSeconds.iloc[0],63060)
        d.loc[3,"AskTime"]="17:32:00"
        self.assertEqual(choose_anchors(timestamp_flags(d)).AnchorSeconds.iloc[0],63000)

    def test_gap_alone_does_not_make_a_common_snapshot(self):
        d=timestamp_flags(pd.DataFrame({"BidTime":["10:00:00","17:30:00","17:29:00","17:29:00"],
                                       "AskTime":["10:00:00","17:31:00","17:31:00","17:30:00"]}))
        d["AnchorSeconds"]=63000
        self.assertEqual(within_anchor(d,60).tolist(),[False,True,False,True])

    def test_bsm_known_call_and_put_call_identity(self):
        c=float(bsm(100,100,1,.05,0,.2,True))
        p=float(bsm(100,100,1,.05,0,.2,False))
        self.assertAlmostEqual(c,10.450583572185565,places=10)
        self.assertAlmostEqual(c-p,100-100*np.exp(-.05),places=11)

    def test_bsm_rejects_sentinel_iv_and_zero_maturity(self):
        for iv,t in [(-99.99,1),(.2,0)]:
            with self.assertRaises(ValueError):bsm(100,100,t,.05,0,iv,True)

    def pair_fixture(self):
        d=pd.DataFrame({"Date":["2020-01-02"]*2,"Strike":[100000]*2,"StrikePoints":[100]*2,
                        "CallPut":["C","P"],"Spot":[100]*2,"r":[0]*2,"q":[0]*2,"Days":[365]*2,
                        "ReferenceExchange":[999]*2,"Bid":[5,6],"Ask":[6,7],
                        "BidSeconds":[63000,63060],"AskSeconds":[63000,63060]})
        for s in [0,60,300]:d["Policy"+str(s)]=True
        return d

    def test_pair_requires_four_times_and_shared_spot(self):
        d=self.pair_fixture()
        p=pair_panel(d,["Date","Strike"],1e-6)
        self.assertFalse(p.Policy0.iloc[0]);self.assertTrue(p.Policy60.iloc[0])
        self.assertFalse(p.ParityOutside.iloc[0]) # boundary: parity=0, interval [-2,0]
        d.loc[1,"Spot"]=101
        p=pair_panel(d,["Date","Strike"],1e-6)
        self.assertFalse(p.SharedInputs.iloc[0]);self.assertTrue(p.VendorParity.isna().iloc[0])

    def test_duplicate_pair_is_rejected(self):
        d=self.pair_fixture()
        with self.assertRaises(ValueError):pair_panel(pd.concat([d,d.iloc[[0]]]),["Date","Strike"],1e-6)


if __name__=="__main__":unittest.main()

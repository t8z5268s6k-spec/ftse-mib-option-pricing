import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_benchmark_protocol_v2 import partition_groups


class ProtocolV2Tests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([{"Date":"2020-01-02","Expiration":expiry,"Strike":k,"CallPut":side,"Bid":1}
                             for expiry in ["2020-06-19","2020-09-18"]
                             for k in range(100,110) for side in ["C","P"]])

    def test_call_put_groups_stay_together_and_split_restarts_per_expiry(self):
        d=partition_groups(self.sample())
        self.assertTrue(d.groupby(["Date","Expiration","Strike"]).Role.nunique().eq(1).all())
        self.assertEqual(set(d.loc[d.Role.eq("development_control"),"Strike"]),{104,109})
        self.assertEqual(int(d.Role.eq("development_control").sum()),8)

    def test_order_and_price_values_do_not_change_membership(self):
        d=self.sample();a=partition_groups(d)
        d=d.sample(frac=1,random_state=9);d["Bid"]=100000
        b=partition_groups(d)
        keys=["Date","Expiration","Strike","CallPut"]
        pd.testing.assert_series_equal(a.set_index(keys).Role.sort_index(),b.set_index(keys).Role.sort_index())

    def test_thin_term_kept_without_inventing_control_rows(self):
        d=self.sample().query("Strike < 103")
        self.assertTrue(partition_groups(d).Role.eq("fit").all())


if __name__=="__main__":unittest.main()

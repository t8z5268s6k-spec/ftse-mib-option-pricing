import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_heston_v2 import estimate_forward


class HestonV2Tests(unittest.TestCase):
    def sample(self):
        records=[]
        for i,k in enumerate([90,100,110]):
            c=20+(100-k)/2;p=20-(100-k)/2
            for j,(kind,value) in enumerate([("C",c),("P",p)]):
                records.append(dict(Date="2020-01-02",Expiration="2021-01-01",Strike=k,CallPut=kind,
                    Bid=value-1,Ask=value+1,r=0,Days=365,Spot=100,Role="fit",ExportRow=2*i+j+2))
        return pd.DataFrame(records)

    def test_forward_uses_shared_interval_and_explicit_fit_provenance(self):
        r=estimate_forward(self.sample(),3)[0]
        self.assertTrue(r["Available"]);self.assertEqual(r["Forward"],100)
        self.assertAlmostEqual(r["qAlternative"],0)
        self.assertEqual(r["FitPairExportRows"],[2,3,4,5,6,7])

    def test_control_rows_cannot_enter_estimator(self):
        d=self.sample();d.loc[0,"Role"]="development_control";d.loc[0,"Bid"]=1000000
        with self.assertRaises(ValueError):estimate_forward(d,3)

    def test_no_fallback_for_insufficient_or_conflicting_pairs(self):
        r=estimate_forward(self.sample().iloc[:4],3)[0]
        self.assertFalse(r["Available"]);self.assertEqual(r["Reason"],"insufficient_fit_pairs")
        d=self.sample();d.loc[0,["Bid","Ask"]]=[100,102]
        r=estimate_forward(d,3)[0]
        self.assertFalse(r["Available"]);self.assertEqual(r["Reason"],"no_positive_common_interval")


if __name__=="__main__":unittest.main()

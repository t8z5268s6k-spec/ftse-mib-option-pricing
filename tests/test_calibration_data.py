import sys
import unittest
from pathlib import Path
import math

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibration_data import chronological_split,match_pairs,parity_fit


def synthetic_quotes():
    rows=[]
    for i,k in enumerate(np.linspace(80,120,15)):
        difference=100*math.exp(-.04)-k*math.exp(.01)
        put=50.
        for side,value in [('C',put+difference),('P',put)]:
            rows.append(dict(SecurityID=1,Date='2020-01-01',Expiration='2020-12-31',
                Strike=k,OptionStyle=0,ExerciseStyle='E',Currency=814,ContractSize=2.5,
                SourceRow=len(rows)+2,OptionID=len(rows)+1,CallPut=side,UnderlyingMid=100.,
                T=1.,OptionMid=value,Bid=value-.5,Ask=value+.5))
    return pd.DataFrame(rows)


class CalibrationDataTests(unittest.TestCase):
    def test_date_split_keeps_same_day_together_and_orders_time(self):
        dates=np.repeat(pd.date_range('2020-01-01',periods=20),2)
        df=pd.DataFrame({'Date':dates,'OptionID':range(40),'SourceRow':range(2,42)})
        split=chronological_split(df.sample(frac=1,random_state=9))
        self.assertEqual(split.groupby('Date').Split.nunique().max(),1)
        self.assertEqual(split.Split.value_counts().to_dict(),{'train':32,'validation':4,'test':4})
        self.assertLess(split[split.Split=='train'].Date.max(),split[split.Split=='validation'].Date.min())
        self.assertLess(split[split.Split=='validation'].Date.max(),split[split.Split=='test'].Date.min())

    def test_exact_synthetic_parity_recovers_negative_rate_and_dividend(self):
        pairs=match_pairs(synthetic_quotes())
        fit=parity_fit(pairs[pairs.PilotPartition=='calibration'])
        self.assertAlmostEqual(fit['r'],-.01,places=12)
        self.assertAlmostEqual(fit['q'],.04,places=12)
        self.assertTrue(fit['SpreadFeasibleBounded'])
        self.assertLessEqual(fit['FeasibleRLow'],-.01)
        self.assertGreaterEqual(fit['FeasibleRHigh'],-.01)
        self.assertLessEqual(fit['FeasibleQLow'],.04)
        self.assertGreaterEqual(fit['FeasibleQHigh'],.04)

    def test_control_quote_changes_do_not_change_training_carry(self):
        raw=synthetic_quotes()
        pairs=match_pairs(raw)
        expected=parity_fit(pairs[pairs.PilotPartition=='calibration'])
        controls=pairs[pairs.PilotPartition=='control']
        ids=set(controls.SourceRow_C)|set(controls.SourceRow_P)
        raw.loc[raw.SourceRow.isin(ids),['Bid','Ask','OptionMid']]+=1000
        changed=match_pairs(raw)
        self.assertEqual(expected,parity_fit(changed[changed.PilotPartition=='calibration']))

    def test_carry_fit_rejects_control_rows(self):
        with self.assertRaises(ValueError):
            parity_fit(match_pairs(synthetic_quotes()))

    def test_ambiguous_pairs_rejected(self):
        raw=synthetic_quotes()
        with self.assertRaises(ValueError):
            match_pairs(pd.concat([raw,raw.iloc[[0]]],ignore_index=True))

    def test_contract_style_and_date_prevent_wrong_matches(self):
        raw=synthetic_quotes()
        raw.loc[raw.CallPut=='P','OptionStyle']=32
        self.assertTrue(match_pairs(raw).empty)
        raw.loc[raw.CallPut=='P','OptionStyle']=0
        raw.loc[raw.CallPut=='P','Date']='2020-01-02'
        self.assertTrue(match_pairs(raw).empty)


if __name__=='__main__':
    unittest.main()

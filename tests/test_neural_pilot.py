import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_neural_pilot import EarlyStop, metrics


class PilotTests(unittest.TestCase):
    def test_stopping_keeps_best_not_last_and_rejects_nan(self):
        s=EarlyStop(2)
        self.assertEqual(s.update(5.,0),(True,False))
        self.assertEqual(s.update(3.,1),(True,False))
        self.assertEqual(s.update(4.,2),(False,False))
        self.assertEqual(s.update(3.,3),(False,True))
        self.assertEqual((s.epoch,s.best),(1,3.))
        with self.assertRaises(ValueError):s.update(float('nan'),4)

    def test_economic_checks_and_masked_greek_errors(self):
        d=pd.DataFrame(dict(S=[100,100],K=[100,100],T=[1,1],R=[0,0],Q=[0,0],IsCall=[1,0],
            OptionMid=[5,5],Bid=[4,4],Ask=[6,6],NPrice=[5,-1],NDelta=[.5,.1],NGamma=[.01,-.01],
            NTheta=[2,2],Delta=[.5,float('nan')],Gamma=[.01,float('nan')],Theta=[2,float('nan')],
            UseDelta=[True,False],UseGamma=[True,False],UseTheta=[True,False]))
        m=metrics(d,'N')
        self.assertEqual(m['within_quote'],1);self.assertEqual(m['negative_prices'],1)
        self.assertEqual(m['price_bound_violations'],1);self.assertEqual(m['delta_bound_violations'],1)
        self.assertEqual(m['negative_gamma'],1)
        self.assertEqual(m['greek_proxy_errors']['Theta']['mae'],0.)


if __name__=='__main__':unittest.main()

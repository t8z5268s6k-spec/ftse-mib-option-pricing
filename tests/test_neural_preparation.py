import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_neural_pilot import bsm_price_greeks, label_audit, fit_scalers, normalize_greeks, FEATURES


class NeuralPreparationTests(unittest.TestCase):
    def test_greek_units_against_finite_differences(self):
        for call in [True,False]:
            x=[100.,105.,.7,.03,.02,.25,call]
            ref=bsm_price_greeks(*x)
            for key,index,h,sign in [('Delta',0,.001,1),('Vega',5,1e-5,1),('Theta',2,1e-5,-1)]:
                up=x.copy();down=x.copy();up[index]+=h;down[index]-=h
                numerical=sign*(bsm_price_greeks(*up)['Price']-bsm_price_greeks(*down)['Price'])/(2*h)
                self.assertAlmostEqual(float(ref[key]),float(numerical),places=6)
            up=x.copy();down=x.copy();up[0]+=.01;down[0]-=.01
            gamma=(bsm_price_greeks(*up)['Price']-2*ref['Price']+bsm_price_greeks(*down)['Price'])/.01**2
            self.assertAlmostEqual(float(ref['Gamma']),float(gamma),places=6)

    def test_masks_preserve_prices_and_reject_wrong_basis(self):
        values={k:float(v) for k,v in bsm_price_greeks(100,100,1,.03,.02,.2,True).items()}
        row=dict(ImpliedVolatility=.2,UnderlyingLast=100,UnderlyingMid=100,StrikePoints=100,Days=365,
                 r=.03,q=.02,CallPut='C',CalculationPrice='M',Last=values['Price'],OptionMid=values['Price'],
                 **{k:values[k] for k in ['Delta','Gamma','Vega','Theta']})
        d=pd.DataFrame([row.copy() for _ in range(5)])
        d.loc[1,'CalculationPrice']='L';d.loc[2,'ImpliedVolatility']=-99.98999786
        d.loc[3,'OptionMid']+=1;d.loc[4,'UnderlyingMid']+=1
        result=label_audit(d)
        self.assertEqual(len(result),5)
        self.assertEqual(result.UseDelta.tolist(),[True,False,False,False,False])
        self.assertFalse(result.UseVega.any())
        self.assertTrue(np.isnan(result.loc[2,'ImpliedVolatility']))

    def test_fit_only_scalers_and_derivative_chain_rule(self):
        d=pd.DataFrame({k:[1.,3.] for k in FEATURES+['OptionMid']})
        d['Role']='fit';d['S']=[100.,104.];d['T']=[.5,1.5];d['OptionMid']=[10.,18.]
        sc=fit_scalers(d)
        labels=pd.DataFrame(dict(Delta=[.4],Gamma=[.03],Theta=[-2.]))
        g=normalize_greeks(labels,sc)
        self.assertAlmostEqual(g.Delta.iloc[0],.2)
        self.assertAlmostEqual(g.Gamma.iloc[0],.03)
        self.assertAlmostEqual(g.Theta.iloc[0],.25)
        d.loc[1,'Role']='development_control'
        with self.assertRaises(ValueError):fit_scalers(d)


if __name__=='__main__':unittest.main()

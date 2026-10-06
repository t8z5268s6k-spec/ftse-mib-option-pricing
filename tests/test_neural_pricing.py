import copy
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from neural_pricing import FEATURES, PriceNetwork, price_and_greeks, pilot_loss


def example_scalers():
    return dict(mean=dict(zip(FEATURES,[22000,22000,.5,.08,.01,.03,0]),OptionMid=800),
        scale=dict(zip(FEATURES,[3000,4000,.3,.05,.01,.02,1]),OptionMid=900),
        derivative_loss_rms=dict(Delta=1.,Gamma=1.,Theta=1.))


def example_inputs():
    return torch.tensor([[21000,22000,.4,.05,.01,.03,1],
                         [25000,23000,.9,.2,-.005,.04,0]],dtype=torch.float64)


class PricingTests(unittest.TestCase):
    def test_exact_function_sign_units_and_second_derivative(self):
        def polynomial(x):
            return .003*x[:,0]**2 + 2*x[:,0]*x[:,2] + 7*x[:,2]**2 + 11*x[:,3]
        x=example_inputs(); p=price_and_greeks(polynomial,x)
        torch.testing.assert_close(p['Delta'],.006*x[:,0]+2*x[:,2])
        torch.testing.assert_close(p['Gamma'],torch.full((2,),.006,dtype=torch.float64))
        torch.testing.assert_close(p['Theta'],-2*x[:,0]-14*x[:,2])
        torch.testing.assert_close(p['InitialVolSensitivity'],22*torch.sqrt(x[:,3]))
        affine=price_and_greeks(lambda z:3*z[:,0]+2*z[:,2],x)
        torch.testing.assert_close(affine['Gamma'],torch.zeros(2,dtype=torch.float64))

    def test_real_network_finite_differences(self):
        m=PriceNetwork(example_scalers());x=example_inputs(); p=price_and_greeks(m,x)
        for key,index,h,sign in [('Delta',0,.5,1),('Theta',2,1e-5,-1)]:
            up=x.clone();down=x.clone();up[:,index]+=h;down[:,index]-=h
            fd=sign*(m(up)-m(down))/(2*h)
            torch.testing.assert_close(p[key],fd,rtol=1e-5,atol=1e-7)
        up=x.clone();down=x.clone();up[:,0]+=.5;down[:,0]-=.5
        torch.testing.assert_close(p['Gamma'],(m(up)-2*m(x)+m(down))/.5**2,rtol=1e-4,atol=1e-8)
        up=x.clone();down=x.clone();up[:,3]=(x[:,3].sqrt()+1e-5)**2;down[:,3]=(x[:,3].sqrt()-1e-5)**2
        torch.testing.assert_close(p['InitialVolSensitivity'],(m(up)-m(down))/2e-5,rtol=1e-5,atol=1e-5)

    def test_batch_independence_and_identical_initialization(self):
        a=PriceNetwork(example_scalers());b=PriceNetwork(example_scalers());x=example_inputs()
        for key,val in a.state_dict().items():torch.testing.assert_close(val,b.state_dict()[key],rtol=0,atol=0)
        a.train(); whole=price_and_greeks(a,x);a.eval()
        single=price_and_greeks(a,x[:1])
        for key in whole:torch.testing.assert_close(whole[key][:1],single[key],rtol=1e-12,atol=1e-10)

    def test_masked_nan_and_empty_greek_losses(self):
        sc=example_scalers();m=PriceNetwork(sc);p=price_and_greeks(m,example_inputs(),training_graph=True)
        targets={k:v.detach().clone() for k,v in p.items()}
        targets['Price']+=100
        masks={k:torch.tensor([True,False]) for k in ['Delta','Gamma','Theta']}
        for key in masks:targets[key][1]=float('nan')
        loss,parts,count=pilot_loss(p,targets,masks,sc,differential=True)
        loss.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None))
        self.assertEqual(count['price'],2);self.assertEqual(count['Gamma'],1)
        for key in masks:masks[key][:]=False
        _,parts,_=pilot_loss(p,targets,masks,sc,differential=True)
        self.assertEqual(float(parts['Gamma'].detach()),0.)
        with self.assertRaises(ValueError):
            pilot_loss(p,targets,masks,sc,differential=True,weights=dict(price=1,Delta=.5,Gamma=.1,Theta=.2,Vega=.2))

    def test_gamma_loss_reaches_weights_and_optimizer_can_step(self):
        sc=example_scalers();m=PriceNetwork(sc);x=example_inputs()
        p=price_and_greeks(m,x,training_graph=True)
        gamma_loss=(p['Gamma']-.001).square().mean()
        g=torch.autograd.grad(gamma_loss,tuple(m.parameters()),allow_unused=True)
        self.assertGreater(sum(float(v.norm()) for v in g if v is not None),0)
        before=copy.deepcopy(m.state_dict())
        optimizer=torch.optim.Adam(m.parameters(),lr=.001)
        p=price_and_greeks(m,x,training_graph=True)
        targets={k:v.detach()+.1 for k,v in p.items()}
        masks={k:torch.ones(2,dtype=torch.bool) for k in ['Delta','Gamma','Theta']}
        loss,_,_=pilot_loss(p,targets,masks,sc,differential=True)
        optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in m.state_dict().items()))

    def test_validation_and_price_only_loss(self):
        sc=example_scalers();m=PriceNetwork(sc);x=example_inputs()
        loss,parts,count=pilot_loss({'Price':m(x)},{'Price':m(x).detach()+2},{},sc,differential=False)
        self.assertAlmostEqual(float(loss.detach()),(2/900)**2)
        self.assertEqual(set(parts),{'price'});self.assertEqual(count,{'price':2})
        with self.assertRaises(ValueError):m(x[:,:4])
        x[0,6]=.5
        with self.assertRaises(ValueError):m(x)


if __name__=='__main__':unittest.main()

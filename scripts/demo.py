"""Public synthetic demo: no market data, credentials or trained checkpoints."""
import argparse
import json
from pathlib import Path
import torch
from heston_benchmark import HestonParameters, analytic_price
from heston_sobol import sobol_replicate, summarize_replicates
from neural_pricing import PriceNetwork, price_and_greeks, FEATURES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/demo.json'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output path to preserve the existing demo.')
    params=HestonParameters()
    reference=analytic_price(100,100,365,.03,.02,params)
    runs=[sobol_replicate(100,100,365,.03,.02,params,paths=8192,steps_per_year=256,seed=700+i) for i in range(4)]
    stats=summarize_replicates(runs)
    error=abs(stats['call']['price']-reference)
    mc_pass=error<=4*stats['call']['se']+.03
    scalers=dict(mean=dict(zip(FEATURES,[100,100,.5,.04,.03,.02,0]),OptionMid=5),
        scale=dict(zip(FEATURES,[20,20,.4,.02,.02,.01,1]),OptionMid=10),
        derivative_loss_rms=dict(Delta=1.,Gamma=1.,Theta=1.))
    torch.set_num_threads(1)
    model=PriceNetwork(scalers,seed=42)
    x=torch.tensor([[100,100,1,.04,.03,.02,1]],dtype=torch.float64)
    ad=price_and_greeks(model,x)
    up=x.clone();down=x.clone();h=.01;up[:,0]+=h;down[:,0]-=h
    with torch.no_grad():
        fd_delta=(model(up)-model(down))/(2*h)
        fd_gamma=(model(up)-2*model(x)+model(down))/h**2
    delta_error=abs(float(ad['Delta'][0]-fd_delta[0]));gamma_error=abs(float(ad['Gamma'][0]-fd_gamma[0]))
    derivative_pass=delta_error<1e-7 and gamma_error<1e-8
    result=dict(data='synthetic illustrative inputs only',
        heston=dict(analytic_call=reference,mc_call_mean=stats['call']['price'],
            mc_mean_standard_error=stats['call']['se'],paths_per_scramble=8192,
            independent_scrambles=4,total_paths=32768,steps_per_year=256,
            balanced_base_sobol_net=True,diagnostic_pass=mc_pass),
        neural=dict(untrained_network=True,purpose='derivative consistency only; not a price-accuracy result',
            absolute_delta_finite_difference_error=delta_error,
            absolute_gamma_finite_difference_error=gamma_error,diagnostic_pass=derivative_pass))
    assert mc_pass and derivative_pass,'Numerical demo check failed; inspect the environment and diagnostics.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

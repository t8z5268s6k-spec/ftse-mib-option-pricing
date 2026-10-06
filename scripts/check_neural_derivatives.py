"""Finite-difference verification on synthetic inputs, with no market training."""
from pathlib import Path
import hashlib
import json
import platform
import pandas as pd
import torch
from neural_pricing import PriceNetwork, price_and_greeks

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'results/current/neural_preparation_v1'
OUT=ROOT/'results/current/neural_derivative_check_v1'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    protected={str(p.relative_to(ROOT)):digest(p) for p in PREP.iterdir() if p.is_file()}
    config=json.loads((PREP/'protocol.json').read_text())
    assert config['features']==['S','K','T','V0','R','Q','IsCall']
    assert config['network']['hidden_widths']==[128,64,32] and config['network']['activation']=='Softplus'
    assert config['network']['batch_normalization'] is False and config['network']['dropout']==0
    protected.update(config['input_sha256'])
    sc=json.loads((PREP/'fit_scalers.json').read_text())
    torch.set_num_threads(1)
    model=PriceNetwork(sc,seed=42)
    # Chosen synthetic levels, never sampled from holdout prices or labels.
    scenarios=[[17000,19500,46/365,.23,.002,.025],
               [21700,21000,323/365,.026,-.003,.03],
               [26600,26500,63/365,.044,.001,.02],
               [26600,25000,364/365,.044,.001,.02]]
    x=torch.tensor([row+[call] for row in scenarios for call in [0,1]],dtype=torch.float64)
    exact=price_and_greeks(model,x)
    settings={'Delta':dict(step=1.,atol=1e-7,rtol=1e-5),
              'Gamma':dict(step=1.,atol=1e-8,rtol=1e-4),
              'Theta':dict(step=1e-4,atol=1e-4,rtol=1e-5),
              'InitialVolSensitivity':dict(step=1e-5,atol=1e-4,rtol=1e-5)}
    records=[]
    with torch.no_grad():
        for key,setting in settings.items():
            for fraction in [1.,.5]:
                h=setting['step']*fraction
                up=x.clone();down=x.clone()
                if key in ('Delta','Gamma'):
                    up[:,0]+=h;down[:,0]-=h
                elif key=='Theta':
                    up[:,2]+=h;down[:,2]-=h
                else:
                    up[:,3]=(x[:,3].sqrt()+h)**2;down[:,3]=(x[:,3].sqrt()-h)**2
                if key=='Gamma': numerical=(model(up)-2*model(x)+model(down))/h**2
                else: numerical=(model(up)-model(down))/(2*h)*(-1 if key=='Theta' else 1)
                for i,(ad,fd) in enumerate(zip(exact[key].tolist(),numerical.tolist())):
                    tol=setting['atol']+setting['rtol']*abs(ad)
                    records.append(dict(case=i,kind=key,step=h,autodiff=ad,finite_difference=fd,
                                        absolute_error=abs(ad-fd),tolerance=tol,passed=abs(ad-fd)<=tol))
    table=pd.DataFrame(records)
    table.to_csv(OUT/'finite_differences.csv',index=False)
    # Changing batch companions, order or train/eval mode must not change a contract.
    batch_error=0.
    model.train()
    for i in range(len(x)):
        single=price_and_greeks(model,x[i:i+1])
        for key in exact:
            batch_error=max(batch_error,abs(float(single[key][0]-exact[key][i])))
    summary=dict(status='passed' if table.passed.all() and batch_error<1e-9 else 'failed',
        synthetic_contracts=len(x),finite_difference_checks=len(table),passed=int(table.passed.sum()),
        max_error_by_kind=table.groupby('kind').absolute_error.max().to_dict(),
        max_batch_or_mode_difference=batch_error,network_parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__,python_version=platform.python_version(),device='CPU',dtype='float64',
        synthetic_inputs=x.tolist(),tolerances=settings,empirical_training_run=False,
        caveat='Derivative consistency only; no pricing accuracy or economic shape guarantee',
        initial_vol_sensitivity_is_not_vendor_vega=True,
        protected_input_sha256=protected,
        script_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/neural_pricing.py',ROOT/'tests/test_neural_pricing.py']})
    assert all(digest(ROOT/p)==sha for p,sha in protected.items())
    summary['protected_input_files_unchanged']=len(protected)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert summary['status']=='passed'
    print(json.dumps({k:v for k,v in summary.items() if k not in ['protected_input_sha256','synthetic_inputs','tolerances','script_sha256']},indent=2))


if __name__=='__main__':main()

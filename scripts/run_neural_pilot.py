"""One frozen-seed, matched BNN/DML development run; never accesses holdouts."""
from pathlib import Path
import copy
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from neural_pricing import FEATURES, ACTIVE_GREEKS, PriceNetwork, price_and_greeks, pilot_loss

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'results/current/neural_preparation_v1'
OUT=ROOT/'results/current/neural_pilot_v1'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class EarlyStop:
    def __init__(self,patience):
        if patience<1: raise ValueError('Positive patience required')
        self.patience=patience;self.best=math.inf;self.epoch=-1;self.stale=0

    def update(self,value,epoch):
        if not math.isfinite(value): raise ValueError('Nonfinite development loss')
        improved=value<self.best
        if improved:self.best=value;self.epoch=epoch;self.stale=0
        else:self.stale+=1
        return improved,self.stale>=self.patience


def tensors(d):
    return (torch.tensor(d[list(FEATURES)].to_numpy(),dtype=torch.float64),
            {'Price':torch.tensor(d.OptionMid.to_numpy(),dtype=torch.float64),
             **{k:torch.tensor(d[k].to_numpy(),dtype=torch.float64) for k in ACTIVE_GREEKS}},
            {k:torch.tensor(d['Use'+k].to_numpy(),dtype=torch.bool) for k in ACTIVE_GREEKS})


def metrics(d,prefix):
    p=d[prefix+'Price'].to_numpy(); e=p-d.OptionMid.to_numpy()
    ds=d.S*np.exp(-d.Q*d['T']);dk=d.K*np.exp(-d.R*d['T']);call=d.IsCall.eq(1)
    lower=np.maximum(np.where(call,ds-dk,dk-ds),0);upper=np.where(call,ds,dk)
    disc=np.exp(-d.Q*d['T']);delta=d[prefix+'Delta'];gamma=d[prefix+'Gamma']
    delta_lo=np.where(call,0,-disc);delta_hi=np.where(call,disc,0)
    return dict(n=len(d),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))),
        within_quote=int(((p>=d.Bid)&(p<=d.Ask)).sum()),
        negative_prices=int((p < -1e-6).sum()),
        price_bound_violations=int(((p<lower-1e-6)|(p>upper+1e-6)).sum()),
        delta_bound_violations=int(((delta<delta_lo-1e-8)|(delta>delta_hi+1e-8)).sum()),
        negative_gamma=int((gamma < -1e-10).sum()),
        greek_proxy_errors={k:dict(n=int(d['Use'+k].sum()),
            mae=float((d.loc[d['Use'+k],prefix+k]-d.loc[d['Use'+k],k]).abs().mean()),
            rmse=float(np.sqrt(((d.loc[d['Use'+k],prefix+k]-d.loc[d['Use'+k],k])**2).mean())))
            for k in ACTIVE_GREEKS if d['Use'+k].any()})


def train_model(name,fit,control,sc,protocol):
    folder=OUT/name;folder.mkdir()
    cfg=protocol['optimizer'];differential=name=='DML'
    model=PriceNetwork(sc,seed=cfg['seed'])
    torch.save(model.state_dict(),folder/'initial.pt')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['epsilon'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    x,targets,masks=tensors(fit);xc,yc,_=tensors(control)
    stop=EarlyStop(cfg['early_stopping_patience'])
    history=[];started=time.monotonic()
    with torch.no_grad():initial=float(((model(xc)-yc['Price'])/sc['scale']['OptionMid']).square().mean())
    stop.update(initial,0);best_state=copy.deepcopy(model.state_dict())
    history.append(dict(epoch=0,development_price_mse=initial,selected_best=True,order_sha256=None))
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();order=torch.randperm(len(fit),generator=generator);batch_total=0.
        consumed=0;label_counts={k:0 for k in ACTIVE_GREEKS}
        for indices in order.split(cfg['batch_size']):
            pred=price_and_greeks(model,x[indices],training_graph=True) if differential else {'Price':model(x[indices])}
            loss,_,counts=pilot_loss(pred,{k:v[indices] for k,v in targets.items()},
                {k:v[indices] for k,v in masks.items()},sc,differential=differential,weights=protocol['loss_weights'])
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ArithmeticError('Nonfinite parameter gradient')
            optimizer.step();consumed+=len(indices);batch_total+=float(loss.detach())*len(indices)
            if differential:
                for k in ACTIVE_GREEKS:label_counts[k]+=counts[k]
        assert consumed==len(fit)
        if differential:assert all(label_counts[k]==int(masks[k].sum()) for k in ACTIVE_GREEKS)
        model.eval()
        with torch.no_grad():
            dev=float(((model(xc)-yc['Price'])/sc['scale']['OptionMid']).square().mean())
            fit_mse=float(((model(x)-targets['Price'])/sc['scale']['OptionMid']).square().mean())
        better,done=stop.update(dev,epoch)
        if better:best_state=copy.deepcopy(model.state_dict())
        history.append(dict(epoch=epoch,development_price_mse=dev,fit_price_mse=fit_mse,
            batch_size_weighted_training_loss=batch_total/len(fit),selected_best=better,
            order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),price_rows_seen=consumed,
            **{k+'_labels_seen':v for k,v in label_counts.items()}))
        if epoch%25==0 or done:
            pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
            print(f'{name}: epoch {epoch}, selected epoch {stop.epoch}, development RMSE {math.sqrt(stop.best)*sc["scale"]["OptionMid"]:.2f}',flush=True)
        if done:break
    model.load_state_dict(best_state);model.eval()
    torch.save(model.state_dict(),folder/'best.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    info=dict(selected_epoch=stop.epoch,epochs_run=epoch,
        stop_reason='patience' if done else 'max_epochs',best_development_standardized_mse=stop.best,
        elapsed_seconds=time.monotonic()-started,initial_development_standardized_mse=initial)
    (folder/'training_summary.json').write_text(json.dumps(info,indent=2)+'\n')
    return model,info


def main():
    OUT.mkdir(exist_ok=False)
    protocol=json.loads((PREP/'protocol.json').read_text());sc=json.loads((PREP/'fit_scalers.json').read_text())
    verified=json.loads((ROOT/'local_metadata/neural_preparation_v1_verification.json').read_text())
    for file,sha in verified['outputs_sha256'].items():assert digest(PREP/file)==sha
    derivative_check=json.loads((ROOT/'results/current/neural_derivative_check_v1/summary.json').read_text())
    assert derivative_check['status']=='passed'
    for file,sha in derivative_check['script_sha256'].items():assert digest(ROOT/file)==sha
    protected={str(p.relative_to(ROOT)):digest(p) for p in PREP.iterdir() if p.is_file()}
    protected.update(protocol['input_sha256'])
    d=pd.read_csv(PREP/'pilot_rows.csv')
    fit=d[d.Role=='fit'].copy();control=d[d.Role=='development_control'].copy()
    assert len(fit)==664 and len(control)==156 and d.Date.le('2022-05-04').all() and not d.UseVega.any()
    assert d.groupby(['Date','Expiration','K']).Role.nunique().eq(1).all()
    plan=dict(protocol=protocol,protected_sha256=protected,seed_count=1,
        purpose='matched development pilot, no hyperparameter search or chronological evaluation',
        device='CPU',dtype='float64',threads=1,deterministic_algorithms=True,
        early_stopping='strict decrease in development price MSE; initial state is an eligible fallback',
        shape_tolerances=dict(price=1e-6,delta=1e-8,gamma=1e-10),
        source_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/neural_pricing.py']})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    summary={'models':{},'scope':plan['purpose'],'no_chronological_holdout_access':True}
    x,_,_=tensors(d)
    for name in ['BNN','DML']:
        model,training=train_model(name,fit,control,sc,protocol)
        pred=price_and_greeks(model,x)
        assert all(torch.isfinite(v).all() for v in pred.values())
        for key,val in pred.items():d[name+key]=val.numpy()
        summary['models'][name]=dict(training=training,
            fit=metrics(d[d.Role=='fit'],name),development_control=metrics(d[d.Role=='development_control'],name))
        d.to_csv(OUT/'predictions.csv',index=False)
    # Identical initialization and permutations over the shared epoch prefix.
    a=torch.load(OUT/'BNN/initial.pt',weights_only=True);b=torch.load(OUT/'DML/initial.pt',weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    ha=pd.read_csv(OUT/'BNN/history.csv');hb=pd.read_csv(OUT/'DML/history.csv')
    common=min(len(ha),len(hb))
    assert ha.order_sha256.iloc[1:common].equals(hb.order_sha256.iloc[1:common])
    strata=[]
    for name in ['BNN','DML']:
        c=d[d.Role=='development_control']
        groups=[('all',c),*[(str(day),g) for day,g in c.groupby('Date')],
                ('calls',c[c.IsCall==1]),('puts',c[c.IsCall==0]),
                ('low_activity',c[(c.Volume<10)&(c.OpenInterest<50)]),
                ('other_activity',c[~((c.Volume<10)&(c.OpenInterest<50))])]
        for group,part in groups:
            strata.append(dict(model=name,group=group,**{k:v for k,v in metrics(part,name).items() if k!='greek_proxy_errors'}))
    pd.DataFrame(strata).to_csv(OUT/'stratified_metrics.csv',index=False)
    assert all(digest(ROOT/p)==sha for p,sha in protected.items())
    summary.update(identical_initialization_verified=True,shared_permutation_epochs=common-1,
        protected_files_unchanged=len(protected),plan_sha256=digest(OUT/'plan.json'),torch_version=torch.__version__)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

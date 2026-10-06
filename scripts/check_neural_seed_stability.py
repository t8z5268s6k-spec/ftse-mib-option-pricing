"""Repeat the frozen pilot at seeds 43/44; reuse and preserve seed 42."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_neural_pilot as pilot
from neural_pricing import PriceNetwork, price_and_greeks, ACTIVE_GREEKS

ROOT=pilot.ROOT
PREP=pilot.PREP
OLD=ROOT/'results/current/neural_pilot_v1'
OUT=ROOT/'results/current/neural_seed_stability_v1'


def main():
    OUT.mkdir(exist_ok=False)
    original=json.loads((OLD/'plan.json').read_text())
    verified=json.loads((ROOT/'local_metadata/neural_pilot_v1_verification.json').read_text())
    for file,sha in verified['files_sha256'].items():assert pilot.digest(ROOT/file)==sha
    for file,sha in original['source_code_sha256'].items():assert pilot.digest(ROOT/file)==sha
    protected=dict(original['protected_sha256'])
    protected.update({str(p.relative_to(ROOT)):pilot.digest(p) for p in OLD.rglob('*') if p.is_file()})
    plan=dict(seeds=[42,43,44],reused_seed=42,new_seeds=[43,44],
        only_configuration_change='optimizer.seed; seed controls both initialization and minibatch permutations',
        unchanged_protocol=original['protocol'],protected_sha256=protected,
        runner_sha256=original['source_code_sha256'],script_sha256=pilot.digest(Path(__file__)),
        aggregation='arithmetic mean and sample standard deviation across seeds; descriptive only, no best-seed selection')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    sc=json.loads((PREP/'fit_scalers.json').read_text());base=pd.read_csv(PREP/'pilot_rows.csv')
    fit=base[base.Role=='fit'];control=base[base.Role=='development_control']
    x,_,_=pilot.tensors(base)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    all_summaries={42:json.loads((OLD/'summary.json').read_text())}
    all_predictions={42:pd.read_csv(OLD/'predictions.csv')}
    previous_out=pilot.OUT
    try:
        for seed in [43,44]:
            folder=OUT/f'seed_{seed}';folder.mkdir();pilot.OUT=folder
            config=copy.deepcopy(original['protocol']);config['optimizer']['seed']=seed
            check=copy.deepcopy(config);check['optimizer']['seed']=42
            assert check==original['protocol']
            (folder/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
            d=base.copy();result={'models':{},'seed':seed}
            for name in ['BNN','DML']:
                print(f'Starting seed {seed}, {name}',flush=True)
                model,training=pilot.train_model(name,fit,control,sc,config)
                pred=price_and_greeks(model,x)
                assert all(torch.isfinite(v).all() for v in pred.values())
                for key,value in pred.items():d[name+key]=value.numpy()
                result['models'][name]=dict(training=training,fit=pilot.metrics(d[d.Role=='fit'],name),
                    development_control=pilot.metrics(d[d.Role=='development_control'],name))
            a=torch.load(folder/'BNN/initial.pt',weights_only=True)
            b=torch.load(folder/'DML/initial.pt',weights_only=True)
            assert all(torch.equal(a[k],b[k]) for k in a)
            ha=pd.read_csv(folder/'BNN/history.csv');hb=pd.read_csv(folder/'DML/history.csv')
            common=min(len(ha),len(hb))
            assert ha.order_sha256.iloc[1:common].equals(hb.order_sha256.iloc[1:common])
            result['identical_initialization_verified']=True
            result['shared_permutation_epochs']=common-1
            d.to_csv(folder/'predictions.csv',index=False)
            (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            all_summaries[seed]=result;all_predictions[seed]=d
    finally:pilot.OUT=previous_out
    rows=[];strata=[]
    for seed in [42,43,44]:
        d=all_predictions[seed];c=d[d.Role=='development_control']
        pd.testing.assert_frame_equal(d[base.columns].reset_index(drop=True),base.reset_index(drop=True))
        for name in ['BNN','DML']:
            item=all_summaries[seed]['models'][name];m=item['development_control']
            row=dict(seed=seed,model=name,**item['training'],**{k:v for k,v in m.items() if k!='greek_proxy_errors'})
            for k in ACTIVE_GREEKS:
                row[k+'_mae']=m['greek_proxy_errors'][k]['mae']
                row[k+'_rmse']=m['greek_proxy_errors'][k]['rmse']
            rows.append(row)
            groups=[('calls',c[c.IsCall==1]),('puts',c[c.IsCall==0]),
                ('low_activity',c[(c.Volume<10)&(c.OpenInterest<50)]),
                ('other_activity',c[~((c.Volume<10)&(c.OpenInterest<50))])]
            for group,part in groups:
                strata.append(dict(seed=seed,model=name,group=group,**{k:v for k,v in pilot.metrics(part,name).items() if k!='greek_proxy_errors'}))
    table=pd.DataFrame(rows);table.to_csv(OUT/'per_seed_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'stratified_metrics.csv',index=False)
    keys=['mae','rmse','within_quote','negative_prices','price_bound_violations','delta_bound_violations','negative_gamma']+[k+'_mae' for k in ACTIVE_GREEKS]
    aggregate={name:{key:dict(mean=float(g[key].mean()),sample_std=float(g[key].std(ddof=1)),minimum=float(g[key].min()),maximum=float(g[key].max())) for key in keys}
               for name,g in table.groupby('model')}
    variation=control[['ExportRow','Date','CallPut']].copy()
    variation_summary={}
    for name in ['BNN','DML']:
        predictions=np.array([all_predictions[s].loc[all_predictions[s].Role=='development_control',name+'Price'].to_numpy() for s in [42,43,44]])
        variation[name+'PriceSampleStd']=predictions.std(axis=0,ddof=1)
        variation_summary[name]=dict(mean_contract_price_sample_std=float(predictions.std(axis=0,ddof=1).mean()),
            median_contract_price_sample_std=float(np.median(predictions.std(axis=0,ddof=1))))
    variation.to_csv(OUT/'prediction_variation.csv',index=False)
    wide=table.pivot(index='seed',columns='model')
    paired={key:int((wide[key]['DML']<wide[key]['BNN']).sum()) for key in ['mae','rmse']+[k+'_mae' for k in ACTIVE_GREEKS]}
    assert all(pilot.digest(ROOT/p)==sha for p,sha in protected.items())
    summary=dict(seeds=[42,43,44],new_training_runs=4,reused_training_runs=2,
        aggregate=aggregate,paired_dml_lower_error_seed_count=paired,
        prediction_variation=variation_summary,protected_files_unchanged=len(protected),
        caveat='Three seeds on the same development data used for stopping; not a chronological test, statistical significance claim or best-seed search',
        plan_sha256=pilot.digest(OUT/'plan.json'))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(table[['seed','model','selected_epoch','epochs_run','mae','rmse','Delta_mae','Gamma_mae','Theta_mae','negative_prices']].to_string(index=False),flush=True)
    print(json.dumps(dict(paired=paired,price_variation=variation_summary),indent=2),flush=True)


if __name__=='__main__':main()

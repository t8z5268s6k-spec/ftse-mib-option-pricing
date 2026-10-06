"""Actual Heston MC on the same 156 development quotes as the neural pilots.

Keep the first 20k-path scramble as the primary result; eight-scramble means
are explicitly separate. No calibration, neural tuning or holdout access.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
import pandas as pd
from heston_benchmark import HestonParameters
from heston_sobol import sobol_replicate, summarize_replicates
from run_neural_pilot import digest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/current/heston_pilot_v2'
OUT=ROOT/'results/current/mc_neural_comparison_v1'
KEYS=['Date','Expiration','UnderlyingMid','StrikePoints','r','q']


def price_metrics(d,col):
    e=d[col]-d.OptionMid
    return dict(n=len(d),mae=float(e.abs().mean()),rmse=float(np.sqrt((e**2).mean())),
                within_quote=int(d[col].between(d.Bid,d.Ask).sum()))


def main():
    OUT.mkdir(exist_ok=False)
    d=pd.read_csv(SOURCE/'verified_control_predictions.csv')
    nn=pd.read_csv(ROOT/'results/current/neural_pilot_v1/predictions.csv')
    nn=nn[nn.Role=='development_control']
    shared=d.merge(nn[['ExportRow','OptionMid','S','K','T','R','Q','CallPut']],on='ExportRow',validate='one_to_one',suffixes=('','_nn'))
    assert len(shared)==len(d)==156
    for a,b in [('OptionMid','OptionMid_nn'),('UnderlyingMid','S'),('StrikePoints','K'),('r','R'),('q','Q')]:
        assert np.allclose(shared[a],shared[b],rtol=0,atol=1e-10)
    assert shared.CallPut.eq(shared.CallPut_nn).all() and np.allclose(shared.Days/365,shared['T'])
    protected={str(p.relative_to(ROOT)):digest(p) for p in SOURCE.rglob('*') if p.is_file()}
    for directory in ['neural_preparation_v1','neural_pilot_v1','neural_seed_stability_v1','heston_mc_validation_v2']:
        for p in (ROOT/'results/current'/directory).rglob('*'):
            if p.is_file():protected[str(p.relative_to(ROOT))]=digest(p)
    for p in [ROOT/'scripts/heston_sobol.py',ROOT/'scripts/heston_benchmark.py']:
        protected[str(p.relative_to(ROOT))]=digest(p)
    old=json.loads((ROOT/'results/current/heston_mc_validation_v2/summary.json').read_text())
    for name,sha in old['script_sha256'].items():assert digest(ROOT/'scripts'/name)==sha
    groups=list(d.groupby(KEYS,sort=True))
    term_keys=['Date','Expiration','UnderlyingMid','r','q']
    terms=list(d.groupby(term_keys,sort=True).groups)
    terms_index={key:i for i,key in enumerate(terms)}
    plan=dict(rows=156,contract_states=len(groups),term_states=len(terms),paths_per_scramble=20000,
        independent_scrambles=8,primary_steps_per_year=256,refinement_steps_per_year=1024,
        primary_estimate='replicate 0, selected before simulation; 20000 paths per price',
        repeated_mean='mean of all 8 scrambles; 160000 paths per price, separately labelled',
        seed_rule='900000 + 100 * sorted term-state index + replicate; refinement adds 10000',
        common_random_numbers='same term-state/replicate seed across strikes; independent scramble panels across replicate indices',
        numerical_gate='mean error versus analytic <= 4 * mean SE + .03 at normalized spot 100',
        rare_payoff='fewer than 30 positive payoffs in any replicate is a separate warning; no normal/t interval then',
        refinement_rule='only contract states whose mean-price gate fails; preserve primary 256 results unchanged',
        sobol_balance_warning='10000 base points plus their antithetics; base count not a power of two',
        scope='same-date development prices, fixed v2 calibration and carry; no market-price-based exclusion or tuning',
        protected_sha256=protected,script_sha256=digest(Path(__file__)))
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    params={day:json.loads((SOURCE/day/'summary.json').read_text())['parameters'] for day in d.Date.unique()}
    outputs=[];replicates=[];refine=[];started=time.monotonic()
    for case,(key,quotes) in enumerate(groups):
        day,expiry,spot,strike,r,q=key
        assert quotes.Days.nunique()==1
        days=int(quotes.Days.iloc[0]);scale=spot/100
        term=terms_index[(day,expiry,spot,r,q)];model=HestonParameters(**params[day])
        runs=[sobol_replicate(100.,strike/scale,days,r,q,model,paths=20000,steps_per_year=256,
                              seed=900000+100*term+rep) for rep in range(8)]
        stats=summarize_replicates(runs)
        replicates.append(dict(case=case,term=term,date=day,expiry=expiry,spot=spot,strike=strike,
                               days=days,r=r,q=q,parameters=params[day],steps_per_year=256,runs=runs))
        failed=False
        for row in quotes.itertuples():
            kind='call' if row.CallPut=='C' else 'put';est=stats[kind]
            delta=est['price']-row.HestonPrice/scale;gate=abs(delta)<=4*est['se']+.03
            failed=failed or not gate
            outputs.append(dict(ExportRow=row.ExportRow,case=case,term=term,
                MC20kPrice=runs[0][kind]*scale,MC160kMean=est['price']*scale,
                MCMeanSE=est['se']*scale,MCSingleRunSampleStd=est['se']*math.sqrt(8)*scale,
                CI95Low=None if est['approximate_ci95_low'] is None else est['approximate_ci95_low']*scale,
                CI95High=None if est['approximate_ci95_high'] is None else est['approximate_ci95_high']*scale,
                MeanAnalyticError=delta*scale,NormalizedMeanAnalyticError=delta,
                NumericalGate=gate,SparsePayoffWarning=est['sparse_payoff_warning'],
                MinPositivePayoffs=min(x['nonzero_'+kind] for x in runs),
                MartingaleError=(stats['discounted_spot']['price']-100*math.exp(-q*days/365))*scale,
                MartingaleSE=stats['discounted_spot']['se']*scale,
                MartingaleGate=abs(stats['discounted_spot']['price']-100*math.exp(-q*days/365))<=4*stats['discounted_spot']['se']+.03,
                ActualSteps=runs[0]['steps']))
        if failed:refine.append((case,quotes,key,term))
        if (case+1)%10==0 or case+1==len(groups):
            print(f'Completed {case+1}/{len(groups)} MC contract states; {len(refine)} flagged for fixed refinement',flush=True)
            (OUT/'replicates.json').write_text(json.dumps(replicates,indent=2)+'\n')
            pd.DataFrame(outputs).to_csv(OUT/'mc_estimates.csv',index=False)
    refined=[]
    for case,quotes,key,term in refine:
        day,expiry,spot,strike,r,q=key;scale=spot/100;days=int(quotes.Days.iloc[0])
        runs=[sobol_replicate(100.,strike/scale,days,r,q,HestonParameters(**params[day]),
            paths=20000,steps_per_year=1024,seed=910000+100*term+rep) for rep in range(8)]
        stats=summarize_replicates(runs)
        refined.append(dict(case=case,steps_per_year=1024,runs=runs,checks=[
            dict(ExportRow=int(row.ExportRow),mean=stats['call' if row.CallPut=='C' else 'put']['price']*scale,
                 se=stats['call' if row.CallPut=='C' else 'put']['se']*scale,
                 analytic=float(row.HestonPrice),passed=abs(stats['call' if row.CallPut=='C' else 'put']['price']-row.HestonPrice/scale)
                 <=4*stats['call' if row.CallPut=='C' else 'put']['se']+.03)
            for row in quotes.itertuples()]))
        print(f'Completed refinement for case {case}',flush=True)
    (OUT/'refinement.json').write_text(json.dumps(refined,indent=2)+'\n')
    results=d.merge(pd.DataFrame(outputs),on='ExportRow',validate='one_to_one')
    results.to_csv(OUT/'comparison.csv',index=False)
    # Each replicate index defines one complete 20k-path prediction panel. No quote-independence assumption.
    panel=[]
    for rep in range(8):
        byrow={}
        for group in replicates:
            part=results[results['case']==group['case']]
            for row in part.itertuples():byrow[row.ExportRow]=group['runs'][rep]['call' if row.CallPut=='C' else 'put']*group['spot']/100
        panel_frame=results.copy();panel_frame['PanelPrice']=panel_frame.ExportRow.map(byrow)
        panel.append(dict(replicate=rep,**price_metrics(panel_frame,'PanelPrice')))
    pd.DataFrame(panel).to_csv(OUT/'replicate_panel_metrics.csv',index=False)
    summaries=[]
    for col,label in [('MC20kPrice','Heston MC: fixed first 20k-path scramble'),('MC160kMean','Heston MC: mean of eight 20k-path scrambles'),('HestonPrice','Analytic Heston reference')]:
        summaries.append(dict(model=label,**price_metrics(results,col)))
    neural=pd.read_csv(ROOT/'results/current/neural_seed_stability_v1/per_seed_metrics.csv')
    for name,g in neural.groupby('model'):
        summaries.append(dict(model=f'{name}: mean of three development runs',n=156,
            mae=float(g.mae.mean()),rmse=float(g.rmse.mean()),within_quote=float(g.within_quote.mean())))
    pd.DataFrame(summaries).to_csv(OUT/'model_comparison.csv',index=False)
    summary=dict(comparison=summaries,rows=156,contract_states=len(groups),term_states=len(terms),
        numerical_gate_passed=int(results.NumericalGate.sum()),sparse_payoff_warnings=int(results.SparsePayoffWarning.sum()),
        martingale_gate_passed_rows=int(results.MartingaleGate.sum()),
        max_abs_mean_analytic_error_points=float(results.MeanAnalyticError.abs().max()),
        median_mean_se_points=float(results.MCMeanSE.median()),max_mean_se_points=float(results.MCMeanSE.max()),
        refinement_contract_states=len(refined),refinement_rows=sum(len(g['checks']) for g in refined),
        refinement_passed_rows=sum(int(c['passed']) for g in refined for c in g['checks']),
        mc20k_panel_mae_range=[min(x['mae'] for x in panel),max(x['mae'] for x in panel)],
        mc20k_panel_rmse_range=[min(x['rmse'] for x in panel),max(x['rmse'] for x in panel)],
        elapsed_seconds=time.monotonic()-started,protected_files_unchanged=len(protected),
        plan_sha256=digest(OUT/'plan.json'),neural_data_used_for_stopping=True,
        limitations='MC uncertainty is sampling-only; shared paths across strikes; eight replicates are not eight market samples; no Vega or chronological test')
    assert all(digest(ROOT/p)==sha for p,sha in protected.items())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

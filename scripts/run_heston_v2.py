"""Run the frozen ten-date Heston pilot; forward sensitivity never refits Heston."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from calibration_helpers import prices, metrics, digest
from heston_benchmark import HestonParameters

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"results/current/benchmark_protocol_v2"
OUT=ROOT/"results/current/heston_pilot_v2"


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def load():
    plan=json.loads((PLAN/"summary.json").read_text())
    proof=json.loads((ROOT/"local_metadata/benchmark_protocol_v2_verification.json").read_text())
    assert digest(PLAN/"summary.json")==proof["summary_sha256"]
    cfgpath=ROOT/"config/benchmark_protocol_v2.json"
    assert digest(cfgpath)==plan["config_sha256"]
    cfg=json.loads(cfgpath.read_text())
    assert digest(ROOT/cfg["source"])==plan["source_sha256"]
    for name,h in plan["outputs_sha256"].items():assert digest(PLAN/name)==h
    member=pd.read_csv(PLAN/"membership.csv")
    raw=pd.read_csv(ROOT/cfg["source"])
    sample=member[["ExportRow","Role"]].merge(raw,on="ExportRow",validate="one_to_one")
    assert len(sample)==plan["rows"] and sample.ExportRow.is_unique
    assert sample.Policy0.all() and sample.Date.between("2019-01-02","2022-05-04").all()
    assert sample.groupby(["Date","Expiration","Strike"]).Role.nunique().eq(1).all()
    sample["StrikeTimes1000"]=sample.Strike
    sample["Strike"]=sample.StrikePoints
    sample["UnderlyingMid"]=sample.Spot
    return cfg,sample


def estimate_forward(fit,minimum_pairs):
    if not fit.Role.eq("fit").all():raise ValueError("Forward estimator accepts fit rows only")
    records=[]
    for expiry,g in fit.groupby("Expiration"):
        if g.Date.nunique()!=1 or g.duplicated(["Strike","CallPut"]).any():raise ValueError("Ambiguous forward fit")
        for key in ["r","Days","Spot"]:
            if g[key].nunique()!=1:raise ValueError("Inconsistent forward inputs")
        p=g[g.CallPut.eq("C")].merge(g[g.CallPut.eq("P")],on="Strike",suffixes=("_C","_P"),validate="one_to_one")
        row={"Date":g.Date.iloc[0],"Expiration":expiry,"FitPairs":len(p),"Available":False,
             "FitPairExportRows":sorted([int(x) for x in p.ExportRow_C]+[int(x) for x in p.ExportRow_P])}
        if len(p)<minimum_pairs:row["Reason"]="insufficient_fit_pairs"
        else:
            ref=g.iloc[0];t=ref.Days/365;D=np.exp(-ref.r*t)
            low=float((p.Strike+(p.Bid_C-p.Ask_P)/D).max())
            high=float((p.Strike+(p.Ask_C-p.Bid_P)/D).min())
            row.update(ForwardLow=low,ForwardHigh=high,IntervalWidth=high-low)
            if high<=0 or low>high:row["Reason"]="no_positive_common_interval"
            else:
                forward=(max(low,0)+high)/2
                row.update(Available=True,Reason="available",Forward=forward,
                           qAlternative=float(ref.r-np.log(forward/ref.Spot)/t))
        records.append(row)
    return records


def bound_check(sample,column,qcolumn="q"):
    ds=sample.Spot*np.exp(-sample[qcolumn]*sample.Days/365)
    dk=sample.Strike*np.exp(-sample.r*sample.Days/365)
    call=sample.CallPut.eq("C")
    low=np.maximum(np.where(call,ds-dk,dk-ds),0)
    high=np.where(call,ds,dk)
    return bool(np.isfinite(sample[column]).all() and sample[column].ge(low-1e-7).all() and sample[column].le(high+1e-7).all())


def run_case(day,cfg,allrows):
    dest=OUT/day;dest.mkdir(exist_ok=True,parents=True)
    if (dest/"summary.json").exists():raise FileExistsError(f"Preserve completed case {day}")
    sample=allrows[allrows.Date.eq(day)].sort_values(["Expiration","Strike","CallPut"]).reset_index(drop=True)
    fit=sample[sample.Role.eq("fit")].copy();check=sample[sample.Role.eq("development_control")]
    assert len(fit) and len(check) and set(fit.ExportRow).isdisjoint(check.ExportRow)
    opt=cfg["calibration"]["optimizer"];started=time.monotonic()
    result={"date":day,"run_date":"2026-10-05","status":"running","attempts":[],
            "plan_sha256":digest(PLAN/"summary.json"),"fit_rows":len(fit),"control_rows":len(check)}
    try:
        # This estimator receives no development-control rows or prices.
        forwards=estimate_forward(fit,cfg["forward_sensitivity"]["minimum_fit_pairs"])
        save(dest/"forward_estimates.json",forwards)
        qmap={r["Expiration"]:r["qAlternative"] for r in forwards if r["Available"]}
        sample["qAlternative"]=sample.Expiration.map(qmap)
        scale=np.maximum((fit.Ask-fit.Bid).to_numpy()/2,1.)
        target=fit.OptionMid.to_numpy()
        def objective(vector):return (prices(fit,vector,opt["integration_order_fit"])-target)/scale
        successful=[]
        for i,start in enumerate(opt["starts"]):
            attempt={"start":i}
            try:
                f=least_squares(objective,start,bounds=(opt["bounds_lower"],opt["bounds_upper"]),
                    x_scale="jac",max_nfev=opt["max_nfev"],ftol=1e-8,xtol=1e-8,gtol=1e-8)
                attempt.update(success=bool(f.success),cost=float(f.cost),nfev=int(f.nfev),message=str(f.message),
                               parameters=asdict(HestonParameters(*f.x)),active_bounds=f.active_mask.tolist(),
                               jacobian_singular_values=np.linalg.svd(f.jac,compute_uv=False).tolist())
                if f.success:successful.append((i,f))
            except (ValueError,RuntimeError,ArithmeticError) as e:attempt.update(success=False,error=str(e))
            result["attempts"].append(attempt);save(dest/"progress.json",result)
            print(f"{day} start {i}: success={attempt['success']}, cost={attempt.get('cost')}, nfev={attempt.get('nfev')}",flush=True)
        if not successful:raise RuntimeError("No successful start; date retained as failure")
        best_id,best=min(successful,key=lambda x:x[1].cost)
        x=best.x;lo=np.array(opt["bounds_lower"]);hi=np.array(opt["bounds_upper"])
        result.update(selected_start=best_id,parameters=asdict(HestonParameters(*x)),
                      near_bounds=(np.minimum((x-lo)/(hi-lo),(hi-x)/(hi-lo))<=1e-6).tolist(),
                      feller_margin=float(2*x[1]*x[2]-x[3]**2))
        sample["HestonPrice"]=prices(sample,x,opt["integration_order_check"])
        sample["FitOrderPrice"]=prices(sample,x,opt["integration_order_fit"])
        sample["FixedHestonPrice"]=prices(sample,list(asdict(HestonParameters()).values()),opt["integration_order_check"])
        sample["ForwardSensitivityPrice"]=np.nan
        sample["ForwardSensitivityFitOrderPrice"]=np.nan
        alt=sample[sample.qAlternative.notna()].copy();alt["q"]=alt.qAlternative
        if len(alt):
            sample.loc[alt.index,"ForwardSensitivityPrice"]=prices(alt,x,opt["integration_order_check"])
            sample.loc[alt.index,"ForwardSensitivityFitOrderPrice"]=prices(alt,x,opt["integration_order_fit"])
        main_delta=float((sample.HestonPrice-sample.FitOrderPrice).abs().max())
        alt_delta=float((sample.ForwardSensitivityPrice-sample.ForwardSensitivityFitOrderPrice).abs().max()) if len(alt) else None
        main_bounds=bound_check(sample,"HestonPrice")
        alt_bounds=bound_check(sample[sample.qAlternative.notna()],"ForwardSensitivityPrice","qAlternative") if len(alt) else True
        sample.to_csv(dest/"predictions.csv",index=False)
        control=sample[sample.Role.eq("development_control")]
        comparable=control[control.qAlternative.notna()]
        result.update(main_quadrature_change=main_delta,sensitivity_quadrature_change=alt_delta,
                      main_bounds_passed=main_bounds,sensitivity_bounds_passed=alt_bounds,
                      metrics={role:{col:metrics(g,col) for col in ["HestonPrice","FixedHestonPrice"]} for role,g in sample.groupby("Role")},
                      forward_sensitivity_control_rows=len(comparable),forward_terms_available=len(qmap),
                      forward_sensitivity_metrics={col:metrics(comparable,col) for col in ["HestonPrice","ForwardSensitivityPrice"]} if len(comparable) else None)
        result["status"]="complete" if main_bounds and main_delta<=.001 and alt_bounds and (alt_delta is None or alt_delta<=.001) else "numerical_check_failed"
    except (ValueError,RuntimeError,ArithmeticError) as e:result.update(status="failed",error=str(e))
    result["elapsed_seconds"]=time.monotonic()-started
    result["outputs_sha256"]={p.name:digest(p) for p in dest.iterdir() if p.suffix==".csv" or p.name=="forward_estimates.json"}
    save(dest/"summary.json",result)
    print(f"{day}: {result['status']} ({result['elapsed_seconds']:.1f}s)",flush=True)


def summarize(cfg,sample):
    cases=[];parts=[]
    for day in sorted(sample.Date.unique()):
        folder=OUT/day;r=json.loads((folder/"summary.json").read_text())
        assert r["plan_sha256"]==digest(PLAN/"summary.json")
        assert all(digest(folder/p)==h for p,h in r["outputs_sha256"].items())
        cases.append(r)
        if (folder/"predictions.csv").exists():
            d=pd.read_csv(folder/"predictions.csv");d["CaseStatus"]=r["status"];parts.append(d)
    full=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    checked=full[full.Role.eq("development_control")].copy() if len(full) else full
    checked.to_csv(OUT/"control_predictions.csv",index=False)
    ok=checked[checked.CaseStatus.eq("complete")].copy() if len(checked) else checked
    summary={"run_date":"2026-10-05","cases":cases,"completed_dates":sum(r["status"]=="complete" for r in cases),
             "failed_or_flagged_dates":[r["date"] for r in cases if r["status"]!="complete"],
             "planned_control_rows":int(sample.Role.eq("development_control").sum()),"numerically_checked_control_rows":len(ok),
             "conditional_same_day_development_only":True,"monte_carlo_run":False,"holdouts_used":False,
             "plan_sha256":digest(PLAN/"summary.json"),"config_sha256":digest(ROOT/"config/benchmark_protocol_v2.json")}
    if len(ok):
        summary["row_metrics"]={c:metrics(ok,c) for c in ["HestonPrice","FixedHestonPrice"]}
        summary["equal_date_metrics"]={}
        for c in ["HestonPrice","FixedHestonPrice"]:
            per=[metrics(g,c) for _,g in ok.groupby("Date")]
            summary["equal_date_metrics"][c]={"dates":len(per),"mae":float(np.mean([m["mae"] for m in per])),
                "rmse":float(np.sqrt(np.mean([m["rmse"]**2 for m in per]))),"within_quote_fraction":float(np.mean([m["within_quote"]/m["n"] for m in per]))}
        sensitivity=ok[ok.qAlternative.notna()]
        summary["forward_sensitivity"]={c:metrics(sensitivity,c) for c in ["HestonPrice","ForwardSensitivityPrice"]} if len(sensitivity) else None
        ok["ActivityFlag"]=(ok.Volume<10)&(ok.OpenInterest<50)
        signed=np.where(ok.CallPut.eq("C"),ok.Spot-ok.Strike,ok.Strike-ok.Spot)/ok.Spot
        ok["Moneyness"]=np.where(abs(ok.Strike/ok.Spot-1)<=.05,"near_spot_5pct",np.where(signed>0,"ITM","OTM"))
        ok["Maturity"]=np.where(ok.Days<90,"30_89_days",np.where(ok.Days<180,"90_179_days","180_365_days"))
        strata=[]
        for field in ["CallPut","ActivityFlag","Moneyness","Maturity"]:
            for label,g in ok.groupby(field):
                for c in ["HestonPrice","FixedHestonPrice"]:
                    strata.append({"Dimension":field,"Group":str(label),"Model":c,**metrics(g,c),
                                   "MeanSpread":float((g.Ask-g.Bid).mean())})
        pd.DataFrame(strata).to_csv(OUT/"stratified_metrics.csv",index=False)
    protected=json.loads((OUT/"protected_before.json").read_text())
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    summary["protected_files_unchanged"]=len(protected)
    summary["outputs_sha256"]={p.name:digest(p) for p in OUT.glob("*.csv")}
    save(OUT/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="cases"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--summarize",action="store_true");parser.add_argument("--dates",nargs="+")
    args=parser.parse_args();cfg,sample=load();OUT.mkdir(exist_ok=True)
    if args.summarize:summarize(cfg,sample)
    else:
        for day in args.dates or sorted(sample.Date.unique()):
            if day not in set(sample.Date):raise ValueError("Date outside frozen plan")
            run_case(day,cfg,sample)

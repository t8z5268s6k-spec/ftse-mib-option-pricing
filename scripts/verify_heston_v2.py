"""Verify saved fits and aggregate all ten dates, retaining numerical warnings."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from calibration_helpers import digest,prices,metrics
from run_heston_v2 import load,bound_check,save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/current/heston_pilot_v2"


def main():
    cfg,source=load();collected=[];daily=[];resolved=[];max_recomputed=0.;max_cost_error=0.;forward_pair_rows=set()
    for day in sorted(source.Date.unique()):
        folder=OUT/day;r=json.loads((folder/"summary.json").read_text())
        assert all(digest(folder/p)==h for p,h in r["outputs_sha256"].items())
        p=pd.read_csv(folder/"predictions.csv").sort_values("ExportRow")
        src=source[source.Date.eq(day)].sort_values("ExportRow")
        assert set(p.ExportRow)==set(src.ExportRow) and p.ExportRow.is_unique
        for c in ["Role","Date","Expiration","CallPut"]:assert p[c].tolist()==src[c].tolist()
        for c in ["Strike","Spot","Bid","Ask","OptionMid","r","q","Days"]:
            np.testing.assert_allclose(p[c],src[c],rtol=0,atol=1e-9)
        fit=p[p.Role.eq("fit")];control=p[p.Role.eq("development_control")]
        assert set(map(tuple,fit[["Expiration","Strike"]].values)).isdisjoint(set(map(tuple,control[["Expiration","Strike"]].values)))
        best=min((a for a in r["attempts"] if a["success"]),key=lambda a:a["cost"])
        assert best["start"]==r["selected_start"] and best["parameters"]==r["parameters"]
        cost=float(np.sum(((fit.FitOrderPrice-fit.OptionMid)/np.maximum((fit.Ask-fit.Bid)/2,1))**2)/2)
        max_cost_error=max(max_cost_error,abs(cost-best["cost"]))
        assert abs(cost-best["cost"])<1e-6
        vector=list(r["parameters"].values())
        recomputed=prices(p,vector,192)
        max_recomputed=max(max_recomputed,float(np.max(np.abs(recomputed-p.HestonPrice))))
        assert bound_check(p,"HestonPrice") and bound_check(p,"FixedHestonPrice")
        for term in json.loads((folder/"forward_estimates.json").read_text()):
            f=fit[fit.Expiration.eq(term["Expiration"])]
            c=f[f.CallPut.eq("C")].set_index("Strike");puts=f[f.CallPut.eq("P")].set_index("Strike")
            strikes=sorted(set(c.index)&set(puts.index))
            assert len(strikes)==term["FitPairs"]
            ids={int(c.loc[k,"ExportRow"]) for k in strikes}|{int(puts.loc[k,"ExportRow"]) for k in strikes}
            assert ids==set(term["FitPairExportRows"]) and ids.isdisjoint(control.ExportRow)
            forward_pair_rows.update(ids)
            if term["Available"]:
                assert len(strikes)>=3
                ref=f.iloc[0];D=math.exp(-ref.r*ref.Days/365)
                low=max(k+(c.loc[k,"Bid"]-puts.loc[k,"Ask"])/D for k in strikes)
                high=min(k+(c.loc[k,"Ask"]-puts.loc[k,"Bid"])/D for k in strikes)
                value=(max(low,0)+high)/2
                assert high>=low and value>0 and abs(value-term["Forward"])<1e-8
                q=ref.r-math.log(value/ref.Spot)/(ref.Days/365)
                assert abs(q-term["qAlternative"])<1e-12
                assert np.allclose(p.loc[p.Expiration.eq(term["Expiration"]),"qAlternative"],q,rtol=0,atol=1e-12)
            else:assert p.loc[p.Expiration.eq(term["Expiration"]),"qAlternative"].isna().all()
        alt=p[p.qAlternative.notna()].copy();alt["q"]=alt.qAlternative
        if len(alt):
            max_recomputed=max(max_recomputed,float(np.max(abs(prices(alt,vector,192)-alt.ForwardSensitivityPrice))))
            assert bound_check(alt,"ForwardSensitivityPrice")
        if r["status"]!="complete":
            a=pd.read_csv(folder/"adaptive_resolution.csv")
            assert set(a.ExportRow)==set(p.ExportRow) and a.Difference192.le(.001).all()
            reference=a.set_index("ExportRow")
            assert np.allclose(p.HestonPrice,p.ExportRow.map(reference.Price192),atol=1e-10,rtol=0)
            # CSV round trips of large prices can move their tiny difference by a few ulps.
            assert np.allclose((a.AdaptivePrice-a.Price192).abs(),a.Difference192,atol=1e-10,rtol=0)
            resolved.append({"Date":day,"original_status":r["status"],"difference96_192":r["main_quadrature_change"],
                             "difference192_adaptive":float(a.Difference192.max()),"parameters_changed":False})
        daily.append({"Date":day,"FitRows":len(fit),"ControlRows":len(control),**r["parameters"],
                      "NearBounds":",".join(k for k,b in zip(r["parameters"],r["near_bounds"]) if b),
                      "MainMAE":metrics(control,"HestonPrice")["mae"],"MainRMSE":metrics(control,"HestonPrice")["rmse"],
                      "MainWithinQuote":metrics(control,"HestonPrice")["within_quote"],
                      "FixedMAE":metrics(control,"FixedHestonPrice")["mae"],
                      "SensitivityControlRows":int(control.qAlternative.notna().sum())})
        collected.append(control)
    assert max_recomputed<1e-8
    check=pd.concat(collected,ignore_index=True);assert len(check)==156
    check.to_csv(OUT/"verified_control_predictions.csv",index=False)
    pd.DataFrame(daily).to_csv(OUT/"verified_daily_results.csv",index=False)
    alt=check[check.qAlternative.notna()]
    equal={}
    for col in ["HestonPrice","FixedHestonPrice"]:
        per=[metrics(g,col) for _,g in check.groupby("Date")]
        equal[col]={"mae":float(np.mean([m["mae"] for m in per])),"rmse":float(np.sqrt(np.mean([m["rmse"]**2 for m in per]))),
                    "within_quote_fraction":float(np.mean([m["within_quote"]/m["n"] for m in per]))}
    check["ActivityFlag"]=(check.Volume<10)&(check.OpenInterest<50)
    signed=np.where(check.CallPut.eq("C"),check.Spot-check.Strike,check.Strike-check.Spot)/check.Spot
    check["Moneyness"]=np.where(abs(check.Strike/check.Spot-1)<=.05,"near_spot_5pct",np.where(signed>0,"ITM","OTM"))
    check["Maturity"]=np.where(check.Days<90,"30_89_days",np.where(check.Days<180,"90_179_days","180_365_days"))
    groups=[]
    for field in ["CallPut","ActivityFlag","Moneyness","Maturity"]:
        for label,g in check.groupby(field):
            for col in ["HestonPrice","FixedHestonPrice"]:
                groups.append({"Dimension":field,"Group":str(label),"Model":col,**metrics(g,col),"MeanSpread":float((g.Ask-g.Bid).mean())})
    pd.DataFrame(groups).to_csv(OUT/"verified_stratified_metrics.csv",index=False)
    protected=json.loads((OUT/"protected_before.json").read_text());assert all(digest(ROOT/p)==h for p,h in protected.items())
    original=json.loads((ROOT/"local_metadata/manifest.json").read_text());assert all(digest(ROOT/f["destination"])==f["sha256"] for f in original["files"])
    summary={"verified_on":"2026-10-05","dates":len(daily),"control_rows":len(check),
        "main":{col:metrics(check,col) for col in ["HestonPrice","FixedHestonPrice"]},"equal_date":equal,
        "forward_sensitivity_same_rows":{col:metrics(alt,col) for col in ["HestonPrice","ForwardSensitivityPrice"]},
        "near_boundary_dates":[d["Date"] for d in daily if d["NearBounds"]],"numerical_warnings_resolved":resolved,
        "forward_fit_quote_rows":len(forward_pair_rows),"forward_control_overlap":False,
        "max_recomputed_price_difference":max_recomputed,"max_saved_objective_difference":max_cost_error,
        "all_integrity_checks_passed":True,"holdouts_used":False,"monte_carlo_run":False,
        "price_recomputation_uses_same_quantlib_engine":True,"original_files_unchanged":len(original["files"]),"protected_files_unchanged":len(protected),
        "raw_run_summary_sha256":digest(OUT/"summary.json"),"outputs_sha256":{p.name:digest(p) for p in OUT.glob("verified_*.csv")}}
    save(OUT/"verified_summary.json",summary)
    save(ROOT/"local_metadata/heston_v2_verification.json",{**summary,"verified_summary_sha256":digest(OUT/"verified_summary.json"),
         "runner_sha256":digest(ROOT/"scripts/run_heston_v2.py"),"verifier_sha256":digest(Path(__file__))})
    print(json.dumps(summary,indent=2));print(pd.DataFrame(daily).to_string(index=False))


if __name__=="__main__":main()

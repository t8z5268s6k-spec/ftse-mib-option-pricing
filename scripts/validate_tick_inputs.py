"""Frozen training-only timestamp policy and vendor input consistency audit.

No Heston fitting, implied-volatility estimation, or parity-based row deletion.
All price diagnostics, including failures, are retained separately from v1 pilots.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from audit_raw_data import sha256
from vendor_carry import attach_vendor_carry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/current/tick_input_validation_v1"


def timestamp_flags(d):
    d = d.copy()
    for side in ["Bid", "Ask"]:
        seconds = pd.to_timedelta(d[side+"Time"], errors="coerce").dt.total_seconds()
        d[side+"Seconds"] = seconds.where(seconds.gt(0) & seconds.lt(86400))
    d["ValidTimes"] = d.BidSeconds.notna() & d.AskSeconds.notna()
    d["GapSeconds"] = (d.BidSeconds-d.AskSeconds).abs()
    return d


def choose_anchors(d):
    same = d[d.ValidTimes & d.GapSeconds.eq(0)]
    freq = same.groupby(["Date","BidSeconds"]).size().rename("Rows").reset_index()
    best = freq.sort_values(["Date","Rows","BidSeconds"], ascending=[True,False,False]).drop_duplicates("Date")
    return best.rename(columns={"BidSeconds":"AnchorSeconds","Rows":"AnchorRows"})


def within_anchor(d, threshold):
    return (d.ValidTimes & d.GapSeconds.le(threshold)
            & (d.BidSeconds-d.AnchorSeconds).abs().le(threshold)
            & (d.AskSeconds-d.AnchorSeconds).abs().le(threshold))


def bsm(s,k,t,r,q,vol,call):
    s,k,t,r,q,vol = [np.asarray(x,dtype=float) for x in (s,k,t,r,q,vol)]
    if not all(np.isfinite(x).all() for x in [s,k,t,r,q,vol]) or any((x<=0).any() for x in [s,k,t,vol]):
        raise ValueError("Require positive finite spot, strike, maturity and IV")
    root = vol*np.sqrt(t)
    d1 = (np.log(s/k)+(r-q+vol*vol/2)*t)/root
    d2 = d1-root
    ds,dk = s*np.exp(-q*t),k*np.exp(-r*t)
    return np.where(call,ds*ndtr(d1)-dk*ndtr(d2),dk*ndtr(-d2)-ds*ndtr(-d1))


def pair_panel(d, keys, tolerance):
    if d.duplicated(keys+["CallPut"]).any():
        raise ValueError("Ambiguous pair membership")
    p = d[d.CallPut.eq("C")].merge(d[d.CallPut.eq("P")],on=keys,suffixes=("_C","_P"),validate="one_to_one")
    p["SharedInputs"] = np.isclose(p.Spot_C,p.Spot_P,rtol=0,atol=tolerance)
    for col in ["r","q","Days","ReferenceExchange"]:
        p["SharedInputs"] &= np.isclose(p[col+"_C"],p[col+"_P"],rtol=0,atol=1e-12)
    times = p[["BidSeconds_C","AskSeconds_C","BidSeconds_P","AskSeconds_P"]]
    p["FourTimeRangeSeconds"] = (times.max(axis=1)-times.min(axis=1)).where(times.notna().all(axis=1))
    p["ParityLow"] = p.Bid_C-p.Ask_P
    p["ParityHigh"] = p.Ask_C-p.Bid_P
    p["VendorParity"] = (p.Spot_C*np.exp(-p.q_C*p.Days_C/365)
                         -p.StrikePoints_C*np.exp(-p.r_C*p.Days_C/365)).where(p.SharedInputs)
    p["ParityDistance"] = np.maximum(np.maximum(p.ParityLow-p.VendorParity,p.VendorParity-p.ParityHigh),0)
    p["ParityOutside"] = p.ParityDistance.gt(tolerance) & p.SharedInputs
    for sec in [0,60,300]:
        p["Policy"+str(sec)] = p["Policy"+str(sec)+"_C"] & p["Policy"+str(sec)+"_P"] & p.FourTimeRangeSeconds.le(sec)
    return p


def count_quotes(d):
    return {"Rows":len(d),"Dates":int(d.Date.nunique()),"Calls":int(d.CallPut.eq("C").sum()),
            "Puts":int(d.CallPut.eq("P").sum()),"Near5":int(d.Near5.sum())}


def diagnostic(d, tolerance):
    v = d[d.ReconstructionAvailable]
    return {**count_quotes(d),"ReconstructionRows":len(v),"MissingReconstruction":len(d)-len(v),
            "AboveTolerance":int(v.AbsLastError.gt(tolerance).sum()),
            "MaxLastError":float(v.AbsLastError.max()) if len(v) else None,
            "MAELast":float(v.AbsLastError.mean()) if len(v) else None,
            "MAEMid":float((v.ProviderPrice-v.OptionMid).abs().mean()) if len(v) else None,
            "QuoteIntervalOutsideBounds":int(d.QuoteIntervalOutsideBounds.sum()),
            "MidOutsideBounds":int(d.MidOutsideBounds.sum())}


def main():
    policy = json.loads((ROOT/"config/tick_quality_policy_v1.json").read_text())
    plan = json.loads((OUT/"plan.json").read_text())
    source = ROOT/"results/current/tick_training_audit_v1/candidate_quotes.csv"
    assert sha256(source)==plan["candidate_sha256"]
    assert sha256(ROOT/"config/tick_quality_policy_v1.json")==plan["policy_sha256"]
    assert sha256(OUT/"selected_dates.csv")==plan["selected_dates_sha256"]
    d = pd.read_csv(source)
    assert d.Date.between(policy["training_start"],policy["training_end"]).all()
    base = timestamp_flags(d[d.EligibleWithCarry & d.SpecialSettlement.eq(0)])
    anchors = choose_anchors(base)
    base = base.merge(anchors,on="Date",how="left",validate="many_to_one")
    for sec in policy["sensitivity_seconds"]:
        base["Policy"+str(sec)] = within_anchor(base,sec)
    anchors.to_csv(OUT/"daily_anchors.csv",index=False)
    coverage=[]
    daily=[]
    for name,mask in [("carry_base",pd.Series(True,index=base.index)),
                      ("equal_within_row",base.GapSeconds.eq(0)),
                      *[("anchor_"+str(s)+"s",base["Policy"+str(s)]) for s in [0,60,300]]]:
        part=base[mask]
        coverage.append({"Policy":name,"Group":"all",**count_quotes(part)})
        for (year,regime),g in part.groupby(["Year","Regime"]):
            coverage.append({"Policy":name,"Group":str(year)+"/"+regime,**count_quotes(g)})
        for day,g in part.groupby("Date"):
            daily.append({"Policy":name,"Date":day,**count_quotes(g),"Expiries":g.Expiration.nunique(),
                          "Strikes":g.StrikePoints.nunique(),"Spots":g.Spot.nunique(),
                          "ReferenceExchanges":g.ReferenceExchange.nunique()})
    pd.DataFrame(coverage).to_csv(OUT/"policy_coverage.csv",index=False)
    pd.DataFrame(daily).to_csv(OUT/"daily_policy_coverage.csv",index=False)
    base[["ExportRow","Date","OptionID","AnchorSeconds","GapSeconds","Policy0","Policy60","Policy300"]].to_csv(OUT/"policy_membership.csv",index=False)
    dates=[x["Date"] for x in plan["dates"]]
    sample=base[base.Date.isin(dates)].copy()
    # SourceRow is a temporary interface alias for the new CSV ExportRow only.
    sample["SourceRow"]=sample.ExportRow
    sample=attach_vendor_carry(sample,pd.read_csv(ROOT/"data/raw/wrds/zero_curve_2018_2023.csv"),
                                pd.read_csv(ROOT/"data/raw/wrds/index_dividend_701057_2018_2023.csv"))
    sample=sample.drop(columns="SourceRow")
    sample["OptionMid"]=(sample.Bid+sample.Ask)/2
    sample["ReconstructionAvailable"]=(np.isfinite(sample.ImpliedVolatility)&sample.ImpliedVolatility.gt(0)
        &np.isfinite(sample.UnderlyingLast)&sample.UnderlyingLast.gt(0)&np.isfinite(sample.Last)&sample.Last.gt(0))
    sample["ProviderPrice"]=np.nan
    v=sample[sample.ReconstructionAvailable]
    sample.loc[v.index,"ProviderPrice"]=bsm(v.UnderlyingLast,v.StrikePoints,v.Days/365,v.r,v.q,v.ImpliedVolatility,v.CallPut.eq("C"))
    sample["AbsLastError"]=(sample.ProviderPrice-sample.Last).abs()
    sample["VendorMidAligned"]=(sample.CalculationPrice.eq("M")&(sample.Last-sample.OptionMid).abs().le(1e-6)
                                 &(sample.UnderlyingLast-sample.Spot).abs().le(1e-6))
    ds=sample.Spot*np.exp(-sample.q*sample.Days/365)
    dk=sample.StrikePoints*np.exp(-sample.r*sample.Days/365)
    sample["LowerBound"]=np.maximum(np.where(sample.CallPut.eq("C"),ds-dk,dk-ds),0)
    sample["UpperBound"]=np.where(sample.CallPut.eq("C"),ds,dk)
    tol=policy["parity_interval_tolerance_points"]
    sample["QuoteIntervalOutsideBounds"]=(sample.Ask.lt(sample.LowerBound-tol)|sample.Bid.gt(sample.UpperBound+tol))
    sample["MidOutsideBounds"]=(sample.OptionMid.lt(sample.LowerBound-tol)|sample.OptionMid.gt(sample.UpperBound+tol))
    sample.to_csv(OUT/"quote_diagnostics.csv",index=False)
    pairs=pair_panel(sample,policy["pair_keys"],tol)
    pairs.to_csv(OUT/"pair_diagnostics.csv",index=False)
    recon=[]; parity=[]
    for scope in ["all_selected", "primary_0s", "sensitivity_60s", "sensitivity_300s"]:
        qmask=pd.Series(True,index=sample.index) if scope=="all_selected" else sample["Policy"+scope.split('_')[-1][:-1]]
        pmask=pd.Series(True,index=pairs.index) if scope=="all_selected" else pairs["Policy"+scope.split('_')[-1][:-1]]
        quotes=sample[qmask]; pp=pairs[pmask]
        for group,g in [("all",quotes),*[(day,quotes[quotes.Date.eq(day)]) for day in sorted(dates)]]:
            recon.append({"Scope":scope,"Group":group,**diagnostic(g,policy["iv_max_absolute_error_gate_points"])})
        for group,g in [("all",pp),*[(day,pp[pp.Date.eq(day)]) for day in sorted(dates)]]:
            shared=g[g.SharedInputs]
            parity.append({"Scope":scope,"Group":group,"Pairs":len(g),"ComparablePairs":len(shared),
                           "DifferentInputs":int((~g.SharedInputs).sum()),"Outside":int(shared.ParityOutside.sum()),
                           "MaxDistance":float(shared.ParityDistance.max()) if len(shared) else None})
    pd.DataFrame(recon).to_csv(OUT/"reconstruction_summary.csv",index=False)
    pd.DataFrame(parity).to_csv(OUT/"parity_summary.csv",index=False)
    sample.groupby(["Date","CalculationPrice","Policy0"],dropna=False).agg(Rows=("ExportRow","size"),
        UsableIV=("ReconstructionAvailable","sum"),MaxError=("AbsLastError","max"),
        MeanError=("AbsLastError","mean")).reset_index().to_csv(OUT/"reconstruction_by_basis.csv",index=False)
    protected=json.loads((OUT/"protected_before.json").read_text())
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    original=json.loads((ROOT/"local_metadata/manifest.json").read_text())
    assert all(sha256(ROOT/f["destination"])==f["sha256"] for f in original["files"])
    gate=[x for x in recon if x["Scope"]=="primary_0s" and x["Group"]!="all"]
    failures=[x["Group"] for x in gate if x["ReconstructionRows"]==0 or x["AboveTolerance"]>0]
    summary={"run_date":"2026-10-04","dates":dates,"coverage":[x for x in coverage if x["Group"]=="all"],
             "reconstruction":[x for x in recon if x["Group"]=="all"],"parity":[x for x in parity if x["Group"]=="all"],
             "primary_reconstruction_failed_dates":failures,"primary_reconstruction_gate_passed":not failures and len(gate)==len(dates),
             "sample_rows":len(sample),"sample_pairs":len(pairs),"models_fitted":False,"temporal_holdouts_used":False,
             "prior_files_unchanged":len(protected),"original_files_unchanged":len(original["files"]),
             "policy_sha256":plan["policy_sha256"],"plan_sha256":sha256(OUT/"plan.json"),
             "output_sha256":{p.name:sha256(p) for p in OUT.glob("*.csv")}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="output_sha256"},indent=2))


if __name__=="__main__":
    main()

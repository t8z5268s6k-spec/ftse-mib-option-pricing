"""Fixed-rate forward interval feasibility on the existing ten-date tick sample.

This checks interval compatibility; no prices, q inputs, or memberships change.
It does not estimate a forecasting input or validate a complete volatility surface.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_raw_data import sha256

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/current/tick_input_validation_v1"
OUT=ROOT/"results/current/forward_consistency_v1"


def intersect_forward(strikes, lows, highs, discount, vendor, tol=1e-6):
    k,lo,hi=[np.asarray(x,dtype=float) for x in [strikes,lows,highs]]
    if not len(k) or not (k.shape==lo.shape==hi.shape):raise ValueError("Need nonempty matching arrays")
    if not all(np.isfinite(x).all() for x in [k,lo,hi]) or not np.isfinite(discount) or discount<=0:
        raise ValueError("Finite intervals and positive discount required")
    if (lo>hi).any() or (k<=0).any() or not np.isfinite(vendor) or vendor<=0:raise ValueError("Invalid interval or forward")
    fl,fu=k+lo/discount,k+hi/discount
    lower,upper=float(fl.max()),float(fu.min())
    relaxed_low=max(lower-tol/discount,0.)
    relaxed_high=upper+tol/discount
    feasible=relaxed_high>0 and relaxed_low<=relaxed_high
    distance=np.maximum(np.maximum(fl-vendor,vendor-fu),0)*discount
    vendor_ok=bool(distance.max()<=tol)
    # A=discount*F. Convex minimax solution for all quote intervals, with A>=0.
    al,au=lower*discount,upper*discount
    minimax_a=max((al+au)/2,0.)
    minimax_error=max(al-minimax_a,minimax_a-au,0.)
    nearest=float(np.clip(vendor,relaxed_low,relaxed_high)) if feasible else None
    if feasible and nearest<=0:nearest=None # positive boundary is open
    return {"CommonForwardLow":lower,"CommonForwardHigh":upper,
            "FeasibleLowWithTolerance":relaxed_low,"FeasibleHighWithTolerance":relaxed_high,
            "ForwardFeasible":bool(feasible),"VendorWithinAll":vendor_ok,
            "VendorOutsidePairs":int((distance>tol).sum()),"MaxVendorParityDistance":float(distance.max()),
            "RawIntersectionWidth":upper-lower,"MinimaxParityDistance":minimax_error,
            "ClosestCompatibleForward":nearest,"ClosestForwardShift":None if nearest is None else nearest-vendor,
            "BindingLowIndex":int(fl.argmax()),"BindingHighIndex":int(fu.argmin())},fl,fu


def main():
    OUT.mkdir(exist_ok=True)
    config=json.loads((ROOT/"config/forward_consistency_v1.json").read_text())
    prior=json.loads((SOURCE/"summary.json").read_text())
    for name in ["quote_diagnostics.csv","pair_diagnostics.csv","selected_dates.csv"]:
        assert sha256(SOURCE/name)==prior["output_sha256"][name]
    q=pd.read_csv(SOURCE/"quote_diagnostics.csv")
    p=pd.read_csv(SOURCE/"pair_diagnostics.csv")
    dates=sorted(prior["dates"])
    assert set(q.Date)==set(dates) and q.Date.le("2022-05-04").all()
    terms=q[["Date","Expiration"]].drop_duplicates().sort_values(["Date","Expiration"])
    records=[];pair_records=[]
    for scope,col in [("all_selected",None),("primary_0s","Policy0"),("sensitivity_60s","Policy60"),("sensitivity_300s","Policy300")]:
        qs=q if col is None else q[q[col]]
        ps=p if col is None else p[p[col]]
        for term in terms.itertuples(index=False):
            g=ps[ps.Date.eq(term.Date)&ps.Expiration.eq(term.Expiration)].copy()
            quotes=qs[qs.Date.eq(term.Date)&qs.Expiration.eq(term.Expiration)]
            reference=q[q.Date.eq(term.Date)&q.Expiration.eq(term.Expiration)]
            for field in ["Spot","r","q","Days","Currency","Exchange","OptionStyle","ExerciseStyle","ContractSize","SpecialSettlement","ReferenceExchange"]:
                assert reference[field].nunique()==1,(term,field)
            ref=reference.iloc[0];t=float(ref.Days)/365;discount=float(np.exp(-ref.r*t))
            vendor=float(ref.Spot*np.exp((ref.r-ref.q)*t))
            row={"Scope":scope,"Date":term.Date,"Expiration":term.Expiration,
                 "Regime":ref.Regime,"Quotes":len(quotes),"Pairs":len(g),"Strikes":int(g.Strike.nunique()),
                 "Spot":float(ref.Spot),"Days":int(ref.Days),"r":float(ref.r),"q":float(ref.q),
                 "Discount":discount,"VendorForward":vendor,"SinglePairOnly":len(g)==1}
            if g.empty:
                row["Status"]="no_pairs";records.append(row);continue
            assert g.SharedInputs.all()
            result,fl,fu=intersect_forward(g.StrikePoints_C,g.ParityLow,g.ParityHigh,discount,vendor,config["parity_tolerance_points"])
            li,ui=result.pop("BindingLowIndex"),result.pop("BindingHighIndex")
            row.update(result)
            row["Status"]=("incompatible_intervals" if not result["ForwardFeasible"] else
                           "vendor_within_common_interval" if result["VendorWithinAll"] else "vendor_outside_common_interval")
            for side,index in [("Low",li),("High",ui)]:
                witness=g.iloc[index]
                row["Binding"+side+"CallExportRow"]=int(witness.ExportRow_C)
                row["Binding"+side+"PutExportRow"]=int(witness.ExportRow_P)
                row["Binding"+side+"Strike"]=float(witness.StrikePoints_C)
            if result["ForwardFeasible"]:
                lower=result["FeasibleLowWithTolerance"];upper=result["FeasibleHighWithTolerance"]
                row["CompatibleQLow"]=float(ref.r-np.log(upper/ref.Spot)/t)
                row["CompatibleQHigh"]=float(ref.r-np.log(lower/ref.Spot)/t) if lower>0 else None
            g["Scope"]=scope;g["ForwardLow"]=fl;g["ForwardHigh"]=fu
            g["VendorForward"]=vendor;g["TermStatus"]=row["Status"]
            pair_records.append(g[["Scope","Date","Expiration","ExportRow_C","ExportRow_P","StrikePoints_C",
                                   "ParityLow","ParityHigh","ForwardLow","ForwardHigh","VendorForward","TermStatus"]])
            records.append(row)
    termframe=pd.DataFrame(records)
    termframe.to_csv(OUT/"term_intervals.csv",index=False)
    pd.concat(pair_records,ignore_index=True).to_csv(OUT/"pair_intervals.csv",index=False)
    bydate=[];totals=[]
    for scope,g in termframe.groupby("Scope"):
        for day,part in [("all",g),*[(day,g[g.Date.eq(day)]) for day in dates]]:
            row={"Scope":scope,"Date":day,"Terms":len(part),"Quotes":int(part.Quotes.sum()),"Pairs":int(part.Pairs.sum()),
                 "NoPairs":int(part.Status.eq("no_pairs").sum()),
                 "VendorFits":int(part.Status.eq("vendor_within_common_interval").sum()),
                 "ForwardShiftPossible":int(part.Status.eq("vendor_outside_common_interval").sum()),
                 "IncompatibleTerms":int(part.Status.eq("incompatible_intervals").sum()),
                 "SinglePairTerms":int(part.SinglePairOnly.sum()),
                 "VendorOutsidePairs":int(part.VendorOutsidePairs.sum()),
                 "OutsidePairsInShiftableTerms":int(part.loc[part.Status.eq("vendor_outside_common_interval"),"VendorOutsidePairs"].sum()),
                 "OutsidePairsInIncompatibleTerms":int(part.loc[part.Status.eq("incompatible_intervals"),"VendorOutsidePairs"].sum())}
            (totals if day=="all" else bydate).append(row)
    pd.DataFrame(bydate).to_csv(OUT/"daily_summary.csv",index=False)
    protected=json.loads((OUT/"protected_before.json").read_text())
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    original=json.loads((ROOT/"local_metadata/manifest.json").read_text())
    assert all(sha256(ROOT/f["destination"])==f["sha256"] for f in original["files"])
    summary={"run_date":"2026-10-05","dates":dates,"totals":totals,
             "interpretation":"conditional interval compatibility with fixed WRDS rates, not a fitted predictive model",
             "carry_overwritten":False,"rows_removed_using_errors":False,"models_fitted":False,"holdouts_used":False,
             "protected_files_unchanged":len(protected),"original_files_unchanged":len(original["files"]),
             "config_sha256":sha256(ROOT/"config/forward_consistency_v1.json"),
             "source_summary_sha256":sha256(SOURCE/"summary.json"),
             "output_sha256":{p.name:sha256(p) for p in OUT.glob("*.csv")}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2))
    print(termframe[termframe.Scope.eq("primary_0s")][["Date","Expiration","Pairs","Status","VendorForward","CommonForwardLow","CommonForwardHigh","MinimaxParityDistance"]].to_string(index=False))


if __name__=="__main__":main()

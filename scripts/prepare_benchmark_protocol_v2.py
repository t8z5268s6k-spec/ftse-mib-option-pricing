"""Freeze a bounded, thesis-aligned tick pilot without running any calibration."""
import json
from pathlib import Path
import pandas as pd
from audit_raw_data import sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/current/benchmark_protocol_v2"


def partition_groups(d):
    keys=["Date","Expiration","Strike"]
    groups=d[keys].drop_duplicates().sort_values(keys).copy()
    groups["StrikeRank"]=groups.groupby(keys[:2]).cumcount()
    groups["Role"]="fit"
    groups.loc[groups.StrikeRank.mod(5).eq(4),"Role"]="development_control"
    return d.merge(groups,on=keys,validate="many_to_one")


def main():
    cfgpath=ROOT/"config/benchmark_protocol_v2.json"
    cfg=json.loads(cfgpath.read_text())
    prior=ROOT/"results/current/tick_input_validation_v1"
    audited=json.loads((prior/"summary.json").read_text())
    source=ROOT/cfg["source"]
    assert sha256(source)==audited["output_sha256"][source.name]
    q=pd.read_csv(source)
    q=q[q.Policy0].copy()
    assert q.Date.between("2019-01-02","2022-05-04").all()
    expected=set(pd.read_csv(ROOT/cfg["dates_source"]).Date)
    assert set(q.Date)==expected and q.ExportRow.is_unique
    d=partition_groups(q)
    assert d.groupby(["Date","Expiration","Strike"]).Role.nunique().eq(1).all()
    assert set(d.loc[d.Role.eq("fit"),"ExportRow"]).isdisjoint(d.loc[d.Role.eq("development_control"),"ExportRow"])
    report=[];terms=[]
    for (day,expiry),g in d.groupby(["Date","Expiration"]):
        fit=g[g.Role.eq("fit")]
        types=fit.groupby("Strike").CallPut.agg(set)
        pairs=int(types.map(lambda x:x=={"C","P"}).sum())
        terms.append({"Date":day,"Expiration":expiry,"FitRows":len(fit),
                      "ControlRows":int(g.Role.eq("development_control").sum()),"FitPairs":pairs,
                      "ForwardPairCountSufficient":pairs>=cfg["forward_sensitivity"]["minimum_fit_pairs"]})
    for day,g in d.groupby("Date"):
        fit=g[g.Role.eq("fit")];control=g[g.Role.eq("development_control")]
        assert len(fit)>0 and len(control)>0
        report.append({"Date":day,"FitRows":len(fit),"ControlRows":len(control),
                       "FitCalls":int(fit.CallPut.eq("C").sum()),"FitPuts":int(fit.CallPut.eq("P").sum()),
                       "ControlCalls":int(control.CallPut.eq("C").sum()),"ControlPuts":int(control.CallPut.eq("P").sum()),
                       "FitExpiries":int(fit.Expiration.nunique()),"FitStrikes":int(fit.Strike.nunique())})
    d[["ExportRow","Date","Expiration","Strike","OptionID","CallPut","StrikeRank","Role"]].to_csv(OUT/"membership.csv",index=False)
    pd.DataFrame(report).to_csv(OUT/"selected_dates.csv",index=False)
    pd.DataFrame(terms).to_csv(OUT/"term_coverage.csv",index=False)
    protected=json.loads((OUT/"protected_before.json").read_text())
    assert all(sha256(ROOT/p)==h for p,h in protected.items())
    summary={"prepared_on":"2026-10-05","version":"2.0","rows":len(d),"dates":len(report),
             "fit_rows":int(d.Role.eq("fit").sum()),"control_rows":int(d.Role.eq("development_control").sum()),
             "control_calls":int((d.Role.eq("development_control")&d.CallPut.eq("C")).sum()),
             "control_puts":int((d.Role.eq("development_control")&d.CallPut.eq("P")).sum()),
             "terms":len(terms),"terms_with_at_least_3_fit_pairs":sum(t["ForwardPairCountSufficient"] for t in terms),
             "group_overlap":False,"models_fitted":False,"forward_values_estimated":False,"holdouts_used":False,
             "protected_files_unchanged":len(protected),"config_sha256":sha256(cfgpath),"source_sha256":sha256(source),
             "outputs_sha256":{p.name:sha256(p) for p in OUT.glob("*.csv")}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2));print(pd.DataFrame(report).to_string(index=False))


if __name__=="__main__":main()

"""Audit a separate training-only WRDS tick export; do not fit or adopt a model.

Candidate flags reproduce the probe's structural rules. Timestamp diagnostics
are reported, not silently used as a new filtering policy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_raw_data import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/current/tick_training_audit_v1"
KEY = ["SecurityID", "Date", "OptionID", "Exchange"]
DIVKEY = ["SecurityID", "Date", "Expiration"]


def counts(d):
    return {"Rows": len(d), "Dates": int(d.Date.nunique()),
            "Calls": int(d.CallPut.eq("C").sum()), "Puts": int(d.CallPut.eq("P").sum()),
            "Near5": int(d.Near5.sum()), "PutDates": int(d.loc[d.CallPut.eq("P"), "Date"].nunique()),
            "Near5Dates": int(d.loc[d.Near5, "Date"].nunique())}


def main():
    manifest = json.loads((OUT / "request_manifest.json").read_text())
    assert sha256(ROOT / manifest["path"]) == manifest["sha256"]
    d = pd.read_csv(ROOT / manifest["path"])
    columns = list(d.columns)
    assert len(d) == manifest["wrds_reported_rows"]
    assert d.Date.between("2019-01-02", "2022-05-04").all()
    assert d.SecurityID.eq(701057).all() and d.Currency.eq(814).all()
    assert d.ExerciseStyle.eq("E").all() and d.CallPut.isin(["C", "P"]).all()
    duplicates = int(d.duplicated(KEY, keep=False).sum())
    assert duplicates == 0 and not d[KEY].isna().any().any()
    d["ExportRow"] = np.arange(2, len(d)+2)
    d["Days"] = (pd.to_datetime(d.Expiration) - pd.to_datetime(d.Date)).dt.days
    d["StrikePoints"] = d.Strike / 1000
    d["Spot"] = (d.UnderlyingBid + d.UnderlyingAsk) / 2
    d["Near5"] = (d.StrikePoints / d.Spot).between(.95, 1.05)
    finite = np.isfinite(d[["Bid", "Ask", "UnderlyingBid", "UnderlyingAsk", "StrikePoints"]]).all(axis=1)
    d["PositiveQuotes"] = d.Bid.gt(0) & d.Ask.ge(d.Bid) & np.isfinite(d[["Bid", "Ask"]]).all(axis=1)
    d["ValidUnexpired"] = (finite & d.PositiveQuotes & d.Days.gt(0) & d.StrikePoints.gt(0)
                           & d.UnderlyingBid.gt(0) & d.UnderlyingAsk.ge(d.UnderlyingBid))
    d["Eligible30to365"] = d.ValidUnexpired & d.Days.between(30, 365)
    div = pd.read_csv(ROOT / "data/raw/wrds/index_dividend_701057_2018_2023.csv")
    rates = pd.read_csv(ROOT / "data/raw/wrds/zero_curve_2018_2023.csv")
    assert not div.groupby(DIVKEY).Rate.nunique(dropna=False).gt(1).any()
    div = div[DIVKEY + ["Rate"]].drop_duplicates().rename(columns={"Rate": "DividendRate"})
    assert not rates.duplicated(["Currency", "Date", "Days"]).any()
    assert np.isfinite(rates[["Rate", "Days"]]).all().all() and rates.Rate.gt(-90).all()
    ranges = rates.groupby(["Currency", "Date"]).Days.agg(CurveMinDays="min", CurveMaxDays="max").reset_index()
    d = d.merge(div, on=DIVKEY, how="left", validate="many_to_one")
    d = d.merge(ranges, on=["Currency", "Date"], how="left", validate="many_to_one")
    sentinel = np.isclose(d.DividendRate, -99.99, atol=1e-5, rtol=0)
    assert not (d.DividendRate.lt(-90) & ~sentinel).any()
    d["ValidDividend"] = np.isfinite(d.DividendRate) & ~sentinel
    d["WithinCurve"] = d.Days.ge(d.CurveMinDays) & d.Days.le(d.CurveMaxDays)
    d["EligibleWithCarry"] = d.Eligible30to365 & d.ValidDividend & d.WithinCurve
    d["Regime"] = np.where(d.Date.lt("2021-12-13"), "pre_2021_12_13", "from_2021_12_13")
    d["Year"] = d.Date.str[:4]
    for side in ["Bid", "Ask", "Last"]:
        seconds = pd.to_timedelta(d[side+"Time"], errors="coerce").dt.total_seconds()
        d[side+"TimeSeconds"] = seconds
        d[side+"TimeValid"] = seconds.gt(0) & seconds.lt(86400)
    d["BothTimesValid"] = d.BidTimeValid & d.AskTimeValid
    d["BidAskTimeGapSeconds"] = (d.BidTimeSeconds-d.AskTimeSeconds).abs().where(d.BothTimesValid)
    stages = []
    for stage, mask in [("raw", pd.Series(True,index=d.index)), ("valid_unexpired",d.ValidUnexpired),
                        ("maturity_30_365",d.Eligible30to365), ("carry_covered_30_365",d.EligibleWithCarry)]:
        p = d[mask]
        stages.append({"Stage":stage,"Group":"all",**counts(p)})
        for field in ["Year", "Regime"]:
            for group,g in p.groupby(field):
                stages.append({"Stage":stage,"Group":group,**counts(g)})
    pd.DataFrame(stages).to_csv(OUT / "coverage_stages.csv", index=False)
    g = d[d.Eligible30to365].copy()
    daily = []
    for day, raw in d.groupby("Date"):
        p = raw[raw.Eligible30to365]
        daily.append({"Date":day,"RawRows":len(raw),**counts(p),
                      "CarryCovered":int(p.EligibleWithCarry.sum()),"Expiries":p.Expiration.nunique(),
                      "Strikes":p.Strike.nunique(), "UnequalTimes":int(p.BidAskTimeGapSeconds.gt(0).sum()),
                      "InvalidTimes":int((~p.BothTimesValid).sum())})
    daily = pd.DataFrame(daily)
    baseline = pd.read_csv(ROOT / "results/current/surface_coverage_audit/raw_training_coverage.csv").groupby("Date").Rows.sum()
    daily["OriginalRawRows"] = daily.Date.map(baseline)
    assert set(daily.Date) == set(baseline.index)
    assert daily.RawRows.eq(daily.OriginalRawRows).all()
    daily.to_csv(OUT / "daily_coverage.csv", index=False)
    # Reconcile all fields on the seven previously exported dates, including missing values.
    probe = json.loads((ROOT / "results/current/wrds_tick_probe_v1/request_manifest.json").read_text())
    matches = []
    for item in probe["downloads"]:
        if item["variant"] != "tick":
            continue
        assert sha256(ROOT / item["path"]) == item["sha256"]
        a = pd.read_csv(ROOT / item["path"]).set_index(KEY).sort_index()
        b = d.loc[d.Date.eq(item["date"]),columns].set_index(KEY).sort_index()
        assert a.index.equals(b.index)
        equal = a.eq(b) | (a.isna() & b.isna())
        assert equal.all().all()
        matches.append({"Date":item["date"],"Rows":len(a),"All35FieldsMatch":True})
    pd.DataFrame(matches).to_csv(OUT / "probe_reconciliation.csv",index=False)
    # Match original clean TRAINING rows only; no validation/test price diagnostics.
    old = pd.read_csv(ROOT / "data/processed/options_clean.csv")
    old = old[old.Date.between("2019-01-02","2022-05-04")].copy()
    old["Days"] = (pd.to_datetime(old.Expiration)-pd.to_datetime(old.Date)).dt.days
    oldkey = ["SecurityID","Date","OptionID"]
    assert not old.duplicated(oldkey).any() and not d.duplicated(oldkey).any()
    joined = old[oldkey+["SourceRow","Days"]].merge(
        d[oldkey+["ExportRow","ValidUnexpired","Eligible30to365"]],on=oldkey,how="left",validate="one_to_one")
    assert joined.ExportRow.notna().all()
    old30 = joined[joined.Days.between(30,365)]
    joined.to_csv(OUT / "original_clean_membership.csv",index=False)
    oldkeys = pd.MultiIndex.from_frame(old30[oldkey])
    g["InOriginalClean30to365"] = pd.MultiIndex.from_frame(g[oldkey]).isin(oldkeys)
    g.to_csv(OUT / "candidate_quotes.csv", index=False)
    g.loc[~g.BothTimesValid | g.BidAskTimeGapSeconds.gt(0)].to_csv(OUT / "timestamp_flags.csv",index=False)
    g.loc[~g.EligibleWithCarry].to_csv(OUT / "carry_exclusions.csv",index=False)
    timestamp = []
    for group,p in [("all",g),*list(g.groupby("Year")),*list(g.groupby("Regime"))]:
        gap = p.BidAskTimeGapSeconds
        timestamp.append({"Group":group,"Rows":len(p),"InvalidTimes":int((~p.BothTimesValid).sum()),
                          "UnequalTimes":int(gap.gt(0).sum()),"GapAbove60s":int(gap.gt(60).sum()),
                          "GapAbove300s":int(gap.gt(300).sum()),"GapAbove3600s":int(gap.gt(3600).sum()),
                          "MaxGapSeconds":float(gap.max()) if gap.notna().any() else None})
    pd.DataFrame(timestamp).to_csv(OUT / "timestamp_summary.csv",index=False)
    d.groupby(["Year","CalculationPrice"],dropna=False).agg(Rows=("OptionID","size"),
        Candidates=("Eligible30to365","sum")).reset_index().to_csv(OUT / "price_basis.csv",index=False)
    original = json.loads((ROOT / "local_metadata/manifest.json").read_text())
    assert all(sha256(ROOT / f["destination"]) == f["sha256"] for f in original["files"])
    protected = json.loads((OUT / "protected_before.json").read_text())
    assert all(sha256(ROOT / p) == expected for p,expected in protected.items())
    summary = {"checked_on":"2026-10-04","raw_rows":len(d),"raw_dates":int(d.Date.nunique()),
               "first_date":d.Date.min(),"last_date":d.Date.max(),"duplicate_key_rows":duplicates,
               "exchange_codes":sorted(map(int,d.Exchange.unique())),"export_columns":columns,
               "coverage":stages,"timestamp":timestamp,"daily_raw_counts_match_original":True,
               "probe_rows_all_fields_match":sum(x["Rows"] for x in matches),
               "original_clean_train_rows":len(old),"original_clean_train_lost_valid_quotes":int((~joined.ValidUnexpired).sum()),
               "original_clean_30_365":len(old30),"old_30_365_lost_eligibility":int((~old30.Eligible30to365).sum()),
               "additional_30_365_candidates":int((~g.InOriginalClean30to365).sum()),
               "candidate_missing_or_sentinel_dividend":int((~g.ValidDividend).sum()),
               "candidate_outside_curve":int((~g.WithinCurve).sum()),
               "candidate_settlement_basis":int(g.CalculationPrice.eq("S").sum()),
               "candidate_underlying_bid_ask_differ":int(g.UnderlyingBid.ne(g.UnderlyingAsk).sum()),
               "candidate_missing_or_nonpositive_iv":int((~np.isfinite(g.ImpliedVolatility) | g.ImpliedVolatility.le(0)).sum()),
               "original_files_unchanged":len(original["files"]),"protected_files_unchanged":len(protected),
               "temporal_holdouts_used":False,"models_fitted":False,"protocol_changed":False,
               "candidate_sample_approved_for_model_training":False,
               "outputs_sha256":{p.name:sha256(p) for p in OUT.glob("*.csv")}}
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["outputs_sha256","export_columns","coverage"]},indent=2))
    print(pd.DataFrame(stages).to_string(index=False))


if __name__ == "__main__":
    main()

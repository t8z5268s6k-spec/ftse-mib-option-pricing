"""Stage 2: explicit cleaning rules and attrition report, without pricing models.

First run audit_raw_data.py. This source-specific workflow uses Strike / 1000.
The output is a quote-quality sample, not a fully validated training dataset.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from audit_raw_data import ROOT, sha256

STRIKE_DIVISOR = 1000.0
SENTINEL_ATOL = 1e-4
GREEKS = ['Delta', 'Gamma', 'Vega', 'Theta']


def clean_candidates(raw):
    """Return all candidates with corrected fields and explicit inclusion flags."""
    df = raw.copy()
    for col in ['Date', 'Expiration', 'StartDate']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in ['Strike', 'Bid', 'Ask', 'UnderlyingBid', 'UnderlyingAsk',
                'UnderlyingLast', 'Last', 'Volume', 'OpenInterest',
                'ContractSize', 'ImpliedVolatility', *GREEKS]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['StrikeRaw'] = df.Strike
    df['Strike'] = df.StrikeRaw / STRIKE_DIVISOR
    for col in ['Bid', 'Ask', 'UnderlyingBid', 'UnderlyingAsk', 'UnderlyingLast',
                'Last', 'Volume', 'OpenInterest', 'ImpliedVolatility', *GREEKS]:
        df[col + 'Raw'] = df[col]
        marker = np.isclose(df[col], -99.99, atol=SENTINEL_ATOL, rtol=0)
        df[col] = df[col].mask(marker | ~np.isfinite(df[col]))
    for col in ['Volume', 'OpenInterest']:
        df[col] = df[col].mask(df[col] < 0)
    df['OptionMid'] = (df.Bid + df.Ask) / 2
    df['UnderlyingMid'] = (df.UnderlyingBid + df.UnderlyingAsk) / 2
    df['T'] = (df.Expiration - df.Date).dt.days / 365.0
    df['is_call'] = df.CallPut.map({'C': 1, 'P': 0}).astype('Int64')
    df['OptionType'] = df.CallPut
    df['BidAskSpread'] = df.Ask - df.Bid
    df['RelativeSpread'] = df.BidAskSpread / df.OptionMid
    # Retain log(S/K) for features; reverse its sign only for put bucket labels.
    with np.errstate(divide='ignore', invalid='ignore'):
        df['log_moneyness'] = np.log(df.UnderlyingMid / df.Strike)
        df['sqrt_T'] = np.sqrt(df['T'].where(df['T'] > 0))
    df['signed_log_moneyness'] = df.log_moneyness.where(df.CallPut == 'C', -df.log_moneyness)
    df['MoneynessBucket'] = pd.cut(df.signed_log_moneyness,
            [-np.inf, -.25, .25, np.inf], labels=['OTM', 'ATM', 'ITM'])
    df['MaturityBucket'] = pd.cut(df['T'], [0, .25, 1, 5, np.inf],
            labels=['Short', 'Medium', 'Long', 'Over5Y'])
    df['IlliquidFlag'] = ((df.Volume < 10) & (df.OpenInterest < 50)).astype('boolean')
    df.loc[df.Volume.isna() | df.OpenInterest.isna(), 'IlliquidFlag'] = pd.NA

    rules = {
        'positive_option_quotes': (df.Bid > 0) & (df.Ask > 0),
        'ordered_option_quotes': df.Ask >= df.Bid,
        'valid_underlying_quotes': (df.UnderlyingBid > 0) & (df.UnderlyingAsk >= df.UnderlyingBid),
        'valid_contract': np.isfinite(df.Strike) & (df.Strike > 0) & df.CallPut.isin(['C', 'P'])
                          & (df.ExerciseStyle == 'E') & df.OptionID.notna() & df.SecurityID.notna(),
        'positive_maturity': np.isfinite(df['T']) & (df['T'] > 0),
        'finite_features': np.isfinite(df[['UnderlyingMid', 'OptionMid', 'Strike', 'T',
                                          'log_moneyness', 'sqrt_T', 'BidAskSpread']]).all(axis=1),
    }
    df['PriceEligible'] = True
    df['FirstRejection'] = ''
    for name, mask in rules.items():
        mask = mask.fillna(False)
        df.loc[df.PriceEligible & ~mask, 'FirstRejection'] = name
        df['PriceEligible'] &= mask
    # Exclude all duplicate keys rather than silently choosing one quote.
    duplicate = df.duplicated(['SecurityID', 'OptionID', 'Date'], keep=False)
    rules['unique_contract_date'] = ~duplicate
    df.loc[df.PriceEligible & duplicate, 'FirstRejection'] = 'unique_contract_date'
    df['PriceEligible'] &= ~duplicate
    df['LockedQuote'] = df.Ask == df.Bid
    # Explicit alternative protocol; a zero-interest exclusion is not base quality cleaning.
    df['ThesisFilterEligible'] = df.PriceEligible & (df.OptionMid > .5) & (df.OpenInterest > 0)
    greek_finite = np.isfinite(df[GREEKS]).all(axis=1)
    delta_sign = ((df.CallPut == 'C') & df.Delta.between(0, 1)) | ((df.CallPut == 'P') & df.Delta.between(-1, 0))
    df['GreekLabelsNumericallyUsable'] = greek_finite & delta_sign & (df.Gamma >= 0) & (df.Vega >= 0)
    df['PositiveIV'] = df.ImpliedVolatility > 0
    # These labels may use a different price convention. No economic consistency claim.
    return df, rules


def describe(df):
    if df.empty:
        return {'rows': 0}
    return {'rows': len(df), 'calls': int((df.CallPut == 'C').sum()),
            'puts': int((df.CallPut == 'P').sum()),
            'date_min': str(df.Date.min().date()), 'date_max': str(df.Date.max().date()),
            'dates': int(df.Date.nunique()), 'contracts': int(df.OptionID.nunique()),
            'volume_zero': int((df.Volume == 0).sum()),
            'open_interest_zero': int((df.OpenInterest == 0).sum()),
            'locked_quotes': int(df.LockedQuote.sum()),
            'illiquid_flag': int(df.IlliquidFlag.fillna(False).sum()),
            'greek_labels_numerically_usable': int(df.GreekLabelsNumericallyUsable.sum()),
            'calculation_price_counts': df.CalculationPrice.value_counts().to_dict(),
            'year_call_put': df.groupby([df.Date.dt.year, 'CallPut']).size().rename('rows').reset_index().to_dict('records'),
            'moneyness_call_put': df.groupby(['CallPut', 'MoneynessBucket'], observed=True).size().rename('rows').reset_index().to_dict('records')}


def main():
    audit_dir = ROOT / 'local_metadata/data_audit'
    profile = json.loads((audit_dir / 'raw_profile.json').read_text())
    if sha256(Path(profile['source'])) != profile['sha256']:
        raise ValueError('Raw workbook differs from the audited input; rerun the full audit.')
    raw = pd.read_csv(audit_dir / 'raw_candidates.csv')
    if len(raw) != profile['candidate_rows']:
        raise ValueError('Candidate count differs from audit profile.')
    df, rules = clean_candidates(raw)
    kept = df.loc[df.PriceEligible].copy().sort_values(['Date', 'OptionID'])
    total = profile['counters']['records']
    attrition = [{'step': 'raw', 'remaining': total, 'removed': 0}]
    attrition.append({'step': 'positive_mid_candidates', 'remaining': len(df), 'removed': total-len(df)})
    mask = pd.Series(True, index=df.index)
    for name, rule in rules.items():
        previous = int(mask.sum())
        mask &= rule.fillna(False)
        attrition.append({'step': name, 'remaining': int(mask.sum()), 'removed': previous-int(mask.sum())})
    assert int(mask.sum()) == len(kept)
    # Independently reproduce the notebook's actual criteria before comparing repairs.
    legacy_spot = (raw.UnderlyingBid + raw.UnderlyingAsk) / 2
    legacy_mid = (raw.Bid + raw.Ask) / 2
    legacy_t = (pd.to_datetime(raw.Expiration) - pd.to_datetime(raw.Date)).dt.days / 365
    legacy = (legacy_t > 1e-6) & (legacy_spot > 0) & (raw.Strike > 0) & (legacy_mid > 0) & (raw.Ask > raw.Bid)
    legacy_bad_leg = legacy & ((raw.Bid <= 0) | (raw.Ask <= 0))
    old_bucket = pd.cut(kept.log_moneyness, [-np.inf, -.25, .25, np.inf], labels=['OTM','ATM','ITM'])
    greek_bad = ~kept.GreekLabelsNumericallyUsable
    stats = {'source_sha256': profile['sha256'], 'candidate_sha256': sha256(audit_dir / 'raw_candidates.csv'),
             'strike_divisor': STRIKE_DIVISOR, 'sentinel_atol': SENTINEL_ATOL,
             'base': describe(kept), 'thesis_filter': describe(kept[kept.ThesisFilterEligible]),
             'legacy_rows': int(legacy.sum()), 'legacy_invalid_quote_legs': int(legacy_bad_leg.sum()),
             'legacy_added_locked': int((df.PriceEligible & ~legacy & df.LockedQuote).sum()),
             'legacy_removed': int((legacy & ~df.PriceEligible).sum()),
             'changed_put_bucket_labels': int(((kept.CallPut == 'P') & (old_bucket != kept.MoneynessBucket)).sum()),
             'missing_greeks_by_field': kept[GREEKS].isna().sum().to_dict(),
             'greek_complete_but_invalid_sign_or_range': int((kept[GREEKS].notna().all(axis=1) & greek_bad).sum()),
             'underlying_bid_equals_ask_all_candidates': bool((raw.UnderlyingBid == raw.UnderlyingAsk).all()),
             'underlying_mid_equals_last_all_candidates': bool((legacy_spot == raw.UnderlyingLast).all()),
             'mid_last_difference_quantiles': (kept.OptionMid-kept.Last).abs().quantile([0,.5,.9,.99,1]).to_dict(),
             'relative_spread_quantiles': kept.RelativeSpread.quantile([0,.25,.5,.75,.9,.99,1]).to_dict(),
             'strike_spot_ratio_before': (raw.Strike / legacy_spot).quantile([0,.5,1]).to_dict(),
             'strike_spot_ratio_after': (kept.Strike / kept.UnderlyingMid).quantile([0,.5,1]).to_dict(),
             'numeric_ranges': kept[['Strike','UnderlyingMid','OptionMid','T',*GREEKS]].agg(['min','median','max']).to_dict(),
             'attrition': attrition, 'models_executed': False}
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out / 'options_clean.csv', index=False)
    df.loc[~df.PriceEligible].to_csv(audit_dir / 'rejected_candidates.csv', index=False)
    df.loc[legacy_bad_leg, ['SourceRow','OptionID','Date','BidRaw','AskRaw','FirstRejection']].to_csv(audit_dir / 'legacy_invalid_quotes.csv', index=False)
    pd.DataFrame(attrition).to_csv(audit_dir / 'filter_counts.csv', index=False)
    stats['output_sha256'] = sha256(out / 'options_clean.csv')
    (audit_dir / 'cleaning_summary.json').write_text(json.dumps(stats, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(stats, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

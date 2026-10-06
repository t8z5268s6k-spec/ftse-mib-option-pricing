"""Prepare matched price/Greek inputs; do not train or read chronological holdouts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/current/heston_pilot_v2'
OUT = ROOT/'results/current/neural_preparation_v1'
FEATURES = ['S', 'K', 'T', 'V0', 'R', 'Q', 'IsCall']
TOLERANCES = dict(Delta=2e-6, Gamma=1e-9, Vega=.01, Theta=.01)


def bsm_price_greeks(s, k, t, r, q, vol, call):
    """European BSM, vol in decimals, calendar-time theta per year."""
    s, k, t, r, q, vol, call = np.broadcast_arrays(s, k, t, r, q, vol, call)
    if not all(np.isfinite(x).all() for x in (s,k,t,r,q,vol)) or np.any((s<=0)|(k<=0)|(t<=0)|(vol<=0)):
        raise ValueError('Finite inputs and positive S/K/T/vol required')
    call = call.astype(bool)
    root = np.sqrt(t)
    d1 = (np.log(s/k)+(r-q+.5*vol**2)*t)/(vol*root)
    d2 = d1-vol*root
    density = np.exp(-.5*d1**2)/np.sqrt(2*np.pi)
    dq, dr = np.exp(-q*t), np.exp(-r*t)
    return dict(Price=np.where(call,s*dq*ndtr(d1)-k*dr*ndtr(d2),k*dr*ndtr(-d2)-s*dq*ndtr(-d1)),
        Delta=dq*np.where(call,ndtr(d1),ndtr(d1)-1), Gamma=dq*density/(s*vol*root),
        Vega=s*dq*density*root,
        Theta=-s*dq*density*vol/(2*root)+np.where(call,
            q*s*dq*ndtr(d1)-r*k*dr*ndtr(d2), -q*s*dq*ndtr(-d1)+r*k*dr*ndtr(-d2)))


def clean_label(values):
    return values.where(np.isfinite(values) & ~np.isclose(values, -99.99, atol=1e-4, rtol=0))


def label_audit(frame):
    d = frame.copy()
    for key in ['ImpliedVolatility','Delta','Gamma','Vega','Theta']:
        d[key] = clean_label(d[key])
    available = d.ImpliedVolatility.gt(0) & d.UnderlyingLast.gt(0) & d.Last.gt(0)
    for key in ['ImpliedVolatility','UnderlyingLast','Last','StrikePoints','Days','r','q']:
        available &= np.isfinite(d[key])
    d['UnitCheckAvailable'] = available
    v = d[available]
    reference = bsm_price_greeks(v.UnderlyingLast,v.StrikePoints,v.Days/365,v.r,v.q,
                                v.ImpliedVolatility,v.CallPut.eq('C'))
    for key, values in reference.items():
        d['BSM'+key] = np.nan
        d.loc[v.index, 'BSM'+key] = values
    d['PriceReconstructionPass'] = available & (d.BSMPrice-d.Last).abs().le(.01)
    d['MidBasisAligned'] = (d.CalculationPrice.eq('M') & (d.Last-d.OptionMid).abs().le(1e-6)
                           & (d.UnderlyingLast-d.UnderlyingMid).abs().le(1e-6))
    for key, tolerance in TOLERANCES.items():
        d[key+'UnitPass'] = available & d[key].notna() & (d[key]-d['BSM'+key]).abs().le(tolerance)
        d['Use'+key] = d.MidBasisAligned & d.PriceReconstructionPass & d[key+'UnitPass']
    # BSM dP/dIV is not dP/d(sqrt(Heston V0)). Keep reference labels, not a false target.
    d['UseVega'] = False
    return d


def fit_scalers(d):
    if not d.Role.eq('fit').all() or len(d) < 2:
        raise ValueError('Scalers require fit rows only')
    values = d[FEATURES+['OptionMid']]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Nonfinite model inputs/target')
    means, scales = values.mean(), values.std(ddof=0)
    scales = scales.mask(scales.eq(0),1.)
    # Binary call/put indicator is deliberately not standardized.
    means['IsCall'], scales['IsCall'] = 0., 1.
    return dict(mean=means.to_dict(), scale=scales.to_dict(), fit_rows=len(d))


def normalize_greeks(d, scalers):
    scales = scalers['scale']
    return pd.DataFrame(dict(Delta=d.Delta*scales['S']/scales['OptionMid'],
        Gamma=d.Gamma*scales['S']**2/scales['OptionMid'],
        Theta=-d.Theta*scales['T']/scales['OptionMid']), index=d.index)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    inputs, frames = {}, []
    for summary_path in sorted(SOURCE.glob('*/summary.json')):
        summary = json.loads(summary_path.read_text())
        file = summary_path.parent/'predictions.csv'
        for path in [summary_path,file]: inputs[str(path.relative_to(ROOT))] = sha(path)
        d = pd.read_csv(file)
        assert d.Date.eq(summary['date']).all() and d.Role.isin(['fit','development_control']).all()
        assert d.Policy0.all() and d.Days.between(30,365).all()
        assert d.Date.le('2022-05-04').all()
        d['V0'] = summary['parameters']['v0']
        d['S'],d['K'],d['T'],d['R'],d['Q'],d['IsCall'] = (
            d.UnderlyingMid,d.StrikePoints,d.Days/365,d.r,d.q,d.CallPut.eq('C').astype(int))
        frames.append(d)
    d = pd.concat(frames,ignore_index=True)
    assert len(d)==820 and d.ExportRow.is_unique
    assert d.groupby(['Date','Expiration','StrikePoints']).Role.nunique().eq(1).all()
    d = label_audit(d)
    fit = d[d.Role=='fit']
    scalers = fit_scalers(fit)
    normalized = normalize_greeks(d,scalers)
    for key in normalized:
        mask = fit['Use'+key]
        values = normalized.loc[fit.index[mask],key]
        scalers.setdefault('derivative_loss_rms',{})[key] = float(max(np.sqrt(np.mean(values**2)),1e-12))
        d['Normalized'+key+'Target'] = normalized[key].where(d['Use'+key])
    columns = ['ExportRow','Date','Expiration','Role','OptionID','CallPut','Days']+FEATURES+[
        'Bid','Ask','OptionMid','Volume','OpenInterest','CalculationPrice','Last','UnderlyingLast',
        'ImpliedVolatility','Delta','Gamma','Vega','Theta','UnitCheckAvailable','PriceReconstructionPass',
        'MidBasisAligned']
    columns += ['BSM'+k for k in ['Price','Delta','Gamma','Vega','Theta']]
    columns += [k+'UnitPass' for k in TOLERANCES]+['Use'+k for k in TOLERANCES]
    columns += ['Normalized'+k+'Target' for k in normalized]
    d[columns].to_csv(OUT/'pilot_rows.csv',index=False)
    (OUT/'fit_scalers.json').write_text(json.dumps(scalers,indent=2)+'\n')
    counts = []
    for role,g in d.groupby('Role'):
        counts.append(dict(role=role,rows=len(g),calls=int(g.IsCall.sum()),puts=int((1-g.IsCall).sum()),
            valid_iv=int(g.UnitCheckAvailable.sum()),mid_aligned=int(g.MidBasisAligned.sum()),
            greek_masks={k:int(g['Use'+k].sum()) for k in TOLERANCES}))
    errors = {k:dict(available=int(d[k+'UnitPass'].notna().sum()),
        checked=int(d.UnitCheckAvailable.sum()),passed=int(d[k+'UnitPass'].sum()),
        max_absolute_error=float((d[k]-d['BSM'+k]).abs().max()),tolerance=tol)
        for k,tol in TOLERANCES.items()}
    # Report raw-label rows with missing IV explicitly; available means reconstruction rows.
    for result in errors.values(): result['available'] = result['checked']
    protocol = dict(version='1.0',date='2026-10-05',scope='preparation for a bounded development pilot; not final empirical training',
        features=FEATURES,variance_input='same-date calibrated Heston v0 from fit rows only, identical for BNN/DML',
        excluded_features=['own ImpliedVolatility','Bid','Ask','BidAskSpread','Last','OptionMid','vendor Greeks'],
        price_target='OptionMid, all 820 rows kept; identical 664 fit and 156 development-control rows for both networks',
        greek_targets='masked vendor BSM proxy sensitivities; Delta/Gamma/Theta only when price/spot basis agrees with mid',
        vega_status='reference only; BSM implied-volatility Vega is not Heston initial-volatility sensitivity; full four-Greek design unresolved',
        loss_weights=dict(price=1.,Delta=.5,Gamma=.1,Theta=.2,Vega=0.),
        normalization='input and price mean/std and derivative RMS from 664 fit rows only; derivative losses averaged over available masks',
        network=dict(hidden_widths=[128,64,32],activation='Softplus',output='one price; Greeks from automatic differentiation',
            batch_normalization=False,dropout=0.,initialization='Xavier uniform',same_architecture_for_bnn_dml=True),
        optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],epsilon=1e-8,batch_size=512,
            seed=42,max_epochs=500,early_stopping_patience=25,selection_metric='development-control price MSE for both models'),
        control_warning='156 rows may select stopping epoch; subsequent scores are development scores, not an untouched test',
        final_holdouts='original chronological boundaries retained and not read; final tick holdout inputs still need preparation',
        unit_tolerances=TOLERANCES,input_sha256=inputs,
        next_step='implement and finite-difference-check one-output neural derivatives before any training')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summary = dict(rows=len(d),counts=counts,unit_checks=errors,
        price_reconstruction_pass=int(d.PriceReconstructionPass.sum()),
        usable_greek_labels_are_proxy_not_observed_market_derivatives=True,
        excluded_vega_labels=int((d.MidBasisAligned & d.PriceReconstructionPass & d.VegaUnitPass).sum()),
        input_files_unchanged=all(sha(ROOT/p)==v for p,v in inputs.items()),
        no_training_run=True,protocol_sha256=sha(OUT/'protocol.json'))
    assert summary['input_files_unchanged']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()

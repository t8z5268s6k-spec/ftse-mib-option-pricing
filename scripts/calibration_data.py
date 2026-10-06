"""Calibration inputs: chronological splits and same-day put-call parity fits.

Rates are option-implied estimates, not imported risk-free/dividend curves.
Pair members are always kept in the same calibration/control partition.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

PAIR_KEYS = ['SecurityID','Date','Expiration','Strike','OptionStyle',
             'ExerciseStyle','Currency','ContractSize']
TERM_KEYS = ['SecurityID','Date','Expiration','OptionStyle','ExerciseStyle','Currency','ContractSize']


def chronological_split(df):
    dates = sorted(pd.to_datetime(df.Date).unique())
    if len(dates) < 10:
        raise ValueError('Need at least ten dates for the 80/10/10 date split.')
    a, b = int(len(dates)*.8), int(len(dates)*.9)
    labels = {day: 'train' if i<a else 'validation' if i<b else 'test' for i,day in enumerate(dates)}
    out = df[['SourceRow','OptionID','Date']].copy()
    out.Date = pd.to_datetime(out.Date)
    out['Split'] = out.Date.map(labels)
    if out.SourceRow.duplicated().any():
        raise ValueError('SourceRow must be unique.')
    return out


def match_pairs(df):
    if df.duplicated(PAIR_KEYS+['CallPut']).any():
        raise ValueError('Ambiguous contract sides; resolve before pairing.')
    out = df[df.CallPut=='C'].merge(df[df.CallPut=='P'],on=PAIR_KEYS,
                                  suffixes=('_C','_P'),validate='one_to_one')
    if not np.allclose(out.UnderlyingMid_C,out.UnderlyingMid_P,rtol=0,atol=1e-8):
        raise ValueError('Call and put underlying references differ.')
    out['ParityMid'] = out.OptionMid_C-out.OptionMid_P
    out['ParityLow'] = out.Bid_C-out.Ask_P
    out['ParityHigh'] = out.Ask_C-out.Bid_P
    out['PilotPartition'] = ''
    for _, group in out.groupby(TERM_KEYS,observed=True):
        ordered = group.sort_values(['Strike','SourceRow_C']).index
        # Every fifth strike held out, selected without looking at prices/errors.
        out.loc[ordered,'PilotPartition'] = ['control' if i%5==0 else 'calibration' for i in range(len(ordered))]
    return out


def parity_fit(training_pairs):
    if len(training_pairs)<6 or training_pairs.Strike.nunique()<6:
        raise ValueError('At least six distinct training strikes required.')
    if (training_pairs.PilotPartition!='calibration').any():
        raise ValueError('Control quotes must not enter the carry fit.')
    if len(training_pairs[TERM_KEYS].drop_duplicates())!=1:
        raise ValueError('Carry fits cannot mix dates, expiries or contract styles.')
    spot=float(training_pairs.UnderlyingMid_C.iloc[0])
    t=float(training_pairs.T_C.iloc[0])
    if t<=0 or (training_pairs.Strike.max()-training_pairs.Strike.min())/spot<.1:
        raise ValueError('Insufficient positive maturity or strike span.')
    x=np.column_stack([np.ones(len(training_pairs)),-training_pairs.Strike.to_numpy()/spot])
    y=training_pairs.ParityMid.to_numpy()/spot
    scale=np.maximum((training_pairs.ParityHigh-training_pairs.ParityLow).to_numpy()/2,1.)/spot
    design=x/scale[:,None]
    a,b=np.linalg.lstsq(design,y/scale,rcond=None)[0]
    if a<=0 or b<=0:
        raise ValueError('Nonpositive implied discount factors.')
    fitted=(x@np.array([a,b]))*spot
    # Project the set of positive discount factors consistent with every training
    # bid/ask parity interval. These are feasible ranges, not statistical CIs.
    matrix=np.vstack([x,-x])
    rhs=np.r_[training_pairs.ParityHigh.to_numpy()/spot,-training_pairs.ParityLow.to_numpy()/spot]
    extremes=[]
    for objective in ([1.,0.],[-1.,0.],[0.,1.],[0.,-1.]):
        result=linprog(objective,A_ub=matrix,b_ub=rhs,bounds=[(1e-9,None),(1e-9,None)],method='highs')
        extremes.append(result)
    ranges={'SpreadFeasibleBounded':all(item.success for item in extremes)}
    if ranges['SpreadFeasibleBounded']:
        a_min,a_max=extremes[0].x[0],extremes[1].x[0]
        b_min,b_max=extremes[2].x[1],extremes[3].x[1]
        ranges.update(FeasibleRLow=float(-np.log(b_max)/t),FeasibleRHigh=float(-np.log(b_min)/t),
                      FeasibleQLow=float(-np.log(a_max)/t),FeasibleQHigh=float(-np.log(a_min)/t))
    return {**ranges,'Spot':spot,'T':t,'DiscountFactor':float(b),'PrepaidForward':float(a*spot),
            'r':float(-np.log(b)/t),'q':float(-np.log(a)/t),'Forward':float(a*spot/b),
            'TrainingPairs':len(training_pairs),'DesignCondition':float(np.linalg.cond(design)),
            'ParityRMSE':float(np.sqrt(np.mean((fitted-training_pairs.ParityMid)**2))),
            'TrainingParityWithinSpread':int(((fitted>=training_pairs.ParityLow)&(fitted<=training_pairs.ParityHigh)).sum())}

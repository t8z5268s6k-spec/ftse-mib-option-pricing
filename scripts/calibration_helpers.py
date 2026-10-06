"""Shared pricing helpers, extracted without the legacy pilot entry point."""
from datetime import date
import hashlib
import numpy as np
from heston_benchmark import HestonParameters, analytic_price


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prices(rows, vector, order=96):
    params=HestonParameters(*vector)
    return np.array([analytic_price(float(row.UnderlyingMid),float(row.Strike),int(row.Days),
        float(row.r),float(row.q),params,row.CallPut,valuation_date=date.fromisoformat(row.Date),
        integration_order=order) for row in rows.itertuples(index=False)])


def metrics(df,column):
    error=df[column]-df.OptionMid
    return {'n':len(df),'mae':float(error.abs().mean()),'rmse':float(np.sqrt((error**2).mean())),
            'within_quote':int(((df[column]>=df.Bid)&(df[column]<=df.Ask)).sum())}

"""Same-date IvyDB Europe carry, with explicit tenor boundaries and provenance."""
import numpy as np
import pandas as pd


def interpolate_rate(curve, days, *, allow_long_flat=False):
    """Interpolate continuous decimal rates, never silently fill missing dates.

    The explicit long flat policy follows the manual's LIBOR-era convention.
    It must not be silently extended to other vendor methodologies.
    """
    c = curve.sort_values("Days")
    if c.empty or c.Days.duplicated().any():
        raise ValueError("Missing or ambiguous curve")
    if not np.isfinite(c[["Days", "Rate"]]).all().all() or c.Days.le(0).any():
        raise ValueError("Invalid curve nodes")
    if c.Rate.lt(-90).any() or not np.isfinite(days) or days <= 0:
        raise ValueError("Invalid rate marker or maturity")
    if days < c.Days.iloc[0]:
        raise ValueError("No short-end extrapolation policy")
    if days > c.Days.iloc[-1]:
        if not allow_long_flat:
            raise ValueError("Long maturity requires explicit policy")
        return float(c.Rate.iloc[-1]), "long_flat"
    return float(np.interp(days, c.Days, c.Rate)), "interpolated_or_exact"


def attach_vendor_carry(sample, rates, dividends, *, allow_libor_long_flat=False):
    keys = ["SecurityID", "Date", "Expiration"]
    if sample.SourceRow.duplicated().any():
        raise ValueError("Duplicate sample rows")
    if dividends.groupby(keys).Rate.nunique(dropna=False).gt(1).any():
        raise ValueError("Conflicting dividend observations")
    unique = dividends[keys + ["Rate"]].drop_duplicates()
    out = sample.drop(columns=["r", "q"], errors="ignore").merge(
        unique.rename(columns={"Rate": "q"}), on=keys, how="left", validate="many_to_one",
    )
    if not np.isfinite(out.q).all() or out.q.lt(-90).any():
        raise ValueError("Missing or invalid dividend input")
    maturity = (pd.to_datetime(out.Expiration) - pd.to_datetime(out.Date)).dt.days
    if not np.array_equal(maturity.to_numpy(), out.Days.to_numpy()):
        raise ValueError("Maturity does not agree with dates")
    curves = {(int(currency), str(day)): g for (currency, day), g in rates.groupby(["Currency", "Date"])}
    records = []
    for row in out.itertuples(index=False):
        curve = curves.get((int(row.Currency), str(row.Date)))
        if curve is None:
            raise ValueError("No same-date curve for sample currency")
        value, policy = interpolate_rate(
            curve, row.Days,
            allow_long_flat=allow_libor_long_flat and row.Date < "2021-12-13",
        )
        records.append((value, policy, int(curve.Days.min()), int(curve.Days.max())))
    out[["r", "RatePolicy", "CurveMinDays", "CurveMaxDays"]] = pd.DataFrame(records, index=out.index)
    out["RateCurveDate"] = out.Date
    out["DividendObservationDate"] = out.Date
    if len(out) != len(sample) or set(out.SourceRow) != set(sample.SourceRow):
        raise ValueError("Carry join changed membership")
    return out

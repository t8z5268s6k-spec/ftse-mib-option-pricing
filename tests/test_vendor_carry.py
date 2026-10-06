import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vendor_carry import interpolate_rate, attach_vendor_carry


class VendorCarryTests(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame({"Date": ["2019-01-01"] * 2, "Currency": [814]*2,
                                   "Days": [7, 365], "Rate": [-.01, .02]})
        self.sample = pd.DataFrame([dict(SourceRow=2, SecurityID=1, Date="2019-01-01",
                                        Expiration="2019-02-01", Days=31, Currency=814)])
        self.div = self.sample[["SecurityID", "Date", "Expiration"]].assign(Rate=.03)

    def test_linear_rate_and_explicit_endpoints(self):
        self.assertAlmostEqual(interpolate_rate(self.rates, 186)[0], .005)
        self.assertEqual(interpolate_rate(self.rates, 7)[0], -.01)
        for day in [1, 366]:
            with self.assertRaises(ValueError): interpolate_rate(self.rates, day)
        self.assertEqual(interpolate_rate(self.rates, 366, allow_long_flat=True), (.02, "long_flat"))

    def test_never_use_future_date_or_other_currency(self):
        for changed in [self.rates.assign(Date="2019-01-02"), self.rates.assign(Currency=1)]:
            with self.assertRaises(ValueError): attach_vendor_carry(self.sample, changed, self.div)

    def test_duplicate_names_safe_but_conflicting_yields_rejected(self):
        result = attach_vendor_carry(self.sample, self.rates, pd.concat([self.div, self.div]))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.q.iloc[0], .03)
        with self.assertRaises(ValueError):
            attach_vendor_carry(self.sample, self.rates, pd.concat([self.div, self.div.assign(Rate=.04)]))

    def test_missing_marker_rejected_and_negative_dividend_preserved(self):
        for invalid in [-99.98999786, float("nan")]:
            with self.assertRaises(ValueError):
                attach_vendor_carry(self.sample, self.rates, self.div.assign(Rate=invalid))
        self.assertEqual(attach_vendor_carry(self.sample, self.rates, self.div.assign(Rate=-.01)).q.iloc[0], -.01)

    def test_maturity_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            attach_vendor_carry(self.sample.assign(Days=32), self.rates, self.div)

    def test_long_flat_not_extended_to_new_vendor_regime(self):
        sample = self.sample.assign(Date="2022-01-01", Expiration="2023-02-01", Days=396)
        div = sample[["SecurityID", "Date", "Expiration"]].assign(Rate=.03)
        with self.assertRaises(ValueError):
            attach_vendor_carry(sample, self.rates.assign(Date="2022-01-01"), div, allow_libor_long_flat=True)


if __name__ == "__main__":
    unittest.main()

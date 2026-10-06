"""Boundary cases that previously corrupted the empirical sample."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_audited_data import clean_candidates


def observation(**changes):
    row = dict(SourceRow=2, SecurityID=701057, OptionID=1, Date='2020-01-01',
               Expiration='2020-07-01', StartDate='2019-01-01', Strike=20000000,
               Bid=100, Ask=110, UnderlyingBid=20000, UnderlyingAsk=20000,
               UnderlyingLast=20000, Last=105, Volume=0, OpenInterest=10,
               ContractSize=2.5, ImpliedVolatility=.2, Delta=.5, Gamma=.001,
               Vega=200, Theta=-20, CallPut='C', ExerciseStyle='E', CalculationPrice='M')
    row.update(changes)
    return row


class DataCleaningTests(unittest.TestCase):
    def clean(self, **changes):
        return clean_candidates(pd.DataFrame([observation(**changes)]))[0].iloc[0]

    def test_positive_average_cannot_rescue_missing_quote(self):
        self.assertGreater((-99.98999786376953 + 200) / 2, 0)
        row = self.clean(Bid=-99.98999786376953, Ask=200)
        self.assertFalse(row.PriceEligible)
        self.assertTrue(pd.isna(row.OptionMid))

    def test_locked_quote_is_valid_and_flagged(self):
        row = self.clean(Bid=100, Ask=100)
        self.assertTrue(row.PriceEligible)
        self.assertTrue(row.LockedQuote)
        self.assertEqual(row.OptionMid, 100)

    def test_expiry_day_and_reversed_market_excluded(self):
        self.assertFalse(self.clean(Expiration='2020-01-01').PriceEligible)
        self.assertFalse(self.clean(Bid=120, Ask=100).PriceEligible)
        self.assertFalse(self.clean(UnderlyingBid=-99.99, UnderlyingAsk=20000).PriceEligible)

    def test_explicit_units_and_call_put_economic_direction(self):
        call = self.clean(UnderlyingBid=30000, UnderlyingAsk=30000)
        put = self.clean(UnderlyingBid=30000, UnderlyingAsk=30000, CallPut='P', Delta=-.5)
        self.assertEqual(call.Strike, 20000)
        self.assertEqual(call.StrikeRaw, 20000000)
        self.assertEqual(call.MoneynessBucket, 'ITM')
        self.assertEqual(put.MoneynessBucket, 'OTM')
        self.assertEqual(call.log_moneyness, put.log_moneyness)

    def test_missing_greek_does_not_remove_price_or_become_zero(self):
        row = self.clean(Delta=-99.98999786376953)
        self.assertTrue(row.PriceEligible)
        self.assertFalse(row.GreekLabelsNumericallyUsable)
        self.assertTrue(pd.isna(row.Delta))
        self.assertEqual(row.Theta, -20)

    def test_zero_activity_retained_and_missing_activity_unknown(self):
        zero = self.clean(Volume=0, OpenInterest=0)
        self.assertTrue(zero.PriceEligible)
        self.assertTrue(zero.IlliquidFlag)
        self.assertFalse(zero.ThesisFilterEligible)
        self.assertTrue(pd.isna(self.clean(OpenInterest=None).IlliquidFlag))

    def test_duplicate_keys_are_not_silently_selected(self):
        df, _ = clean_candidates(pd.DataFrame([observation(), observation(Ask=115)]))
        self.assertFalse(df.PriceEligible.any())
        self.assertTrue((df.FirstRejection == 'unique_contract_date').all())

    def test_unknown_type_is_not_treated_as_put(self):
        row = self.clean(CallPut='X')
        self.assertFalse(row.PriceEligible)
        self.assertTrue(pd.isna(row.is_call))


if __name__ == '__main__':
    unittest.main()

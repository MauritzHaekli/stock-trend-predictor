import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from backend.utils.change_calculator import ChangeCalculator


class TestChangeCalculator(unittest.TestCase):
    def setUp(self):
        self.data = pd.Series([10, 15, 20, 25, 30], name="TestFeature")
        self.rounding_factor = 2

        BaseValidatorMock = MagicMock()
        PeriodValidatorMock = MagicMock()

        BaseValidatorMock.validate_series = MagicMock()
        PeriodValidatorMock.validate_periods = MagicMock()

        ChangeCalculator.base_validator = BaseValidatorMock
        ChangeCalculator.period_validator = PeriodValidatorMock

        self.calculator = ChangeCalculator(self.data, rounding_factor=self.rounding_factor)

    def test_get_absolute_change(self):
        result = self.calculator.get_absolute_change(periods=1)
        expected = pd.Series([np.nan, 5, 5, 5, 5], name="TestFeature(1) change", dtype="float64")
        pd.testing.assert_series_equal(result, expected)

    def test_get_relative_change(self):
        result = self.calculator.get_relative_change(periods=1)
        expected = pd.Series([np.nan, 50.0, 33.33, 25.0, 20.0], name="TestFeature(1) change(%)", dtype="float64")
        pd.testing.assert_series_equal(result, expected)

    def test_get_trend_change(self):
        result = self.calculator.get_trend_change(periods=1)
        expected = pd.Series([0, 1, 1, 1, 1], name="TestFeature(1) trend", dtype="float64")
        pd.testing.assert_series_equal(result, expected)

    def test_get_all_changes(self):
        result = self.calculator.get_all_changes(periods=1)
        expected = pd.DataFrame({
            "TestFeature(1) change": pd.Series([np.nan, 5, 5, 5, 5], dtype="int32"),
            "TestFeature(1) change(%)": pd.Series([np.nan, 50.0, 33.33, 25.0, 20.0], dtype="int32"),
            "TestFeature(1) trend": pd.Series([0, 1, 1, 1, 1], dtype="int32")
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_get_changes_for_periods(self):
        periods_list = [1, 2]
        result = self.calculator.get_changes_for_periods(periods_list)
        expected = pd.DataFrame({
            "TestFeature(1) change": pd.Series([np.nan, 5, 5, 5, 5], dtype="int32"),
            "TestFeature(2) change": pd.Series([np.nan, np.nan, 10, 10, 10], dtype="int32"),
            "TestFeature(1) change(%)": pd.Series([np.nan, 50.0, 33.33, 25.0, 20.0], dtype="int32"),
            "TestFeature(2) change(%)": pd.Series([np.nan, np.nan, 100.0, 66.67, 50.0],dtype="int32"),
            "TestFeature(1) trend": pd.Series([0, 1, 1, 1, 1], dtype="int32"),
            "TestFeature(2) trend": pd.Series([0, 0, 1, 1, 1], dtype="int32"),
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_validator_calls(self):
        self.calculator.get_absolute_change(periods=1)
        ChangeCalculator.base_validator.validate_series.assert_called_once_with(self.data)

        self.calculator.get_relative_change(periods=1)
        ChangeCalculator.period_validator.validate_periods.assert_called_with(1, self.data)

        self.calculator.get_trend_change(periods=1)
        ChangeCalculator.period_validator.validate_periods.assert_called_with(1, self.data)


if __name__ == "__main__":
    unittest.main()

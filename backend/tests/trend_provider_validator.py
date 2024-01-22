import unittest
import pandas as pd
import numpy as np
from backend.utils.trend_provider import TrendProvider


class TestTrendProvider(unittest.TestCase):
    """
    Tests the functionality of the TrendProvider class. It verifies the correct calculation of absolute and percentage changes as well as trend calculations.
    """
    def setUp(self):
        testing_time_series: dict = {
            'datetime': pd.date_range('2024-01-01', '2024-01-10'),
            'open': [103, 102, 117, 93, 105, 92, 109, 104, 118, 91],
            'close': [100, 105, 110, 95, 102, 98, 105, 108, 112, 108]
        }
        self.testing_time_series: pd.DataFrame = pd.DataFrame(testing_time_series)
        self.rounding_factor: int = 4

    def test_get_absolute_change(self):
        column_name = 'close'
        periods = 2
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([np.nan, np.nan, 10.0, -10.0, -8.0, 3.0, 3.0, 10.0, 7.0, 0.0])
        result = trend_provider.get_absolute_change(column_name, periods)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=True, check_names=False)

    def test_get_percentage_change(self):
        column_name = 'close'
        periods = 2
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([np.nan, np.nan, 10.0, -9.5238, -7.2727, 3.1579, 2.9412, 10.2041, 6.6667, 0.0])
        result = trend_provider.get_percentage_change(column_name, periods)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=True, check_names=False)

    def test_get_column_difference(self):
        first_column_name: str = 'open'
        second_column_name: str = 'close'
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([-3, 3, -7, 2, -3, 6, -4, 4, -6, 17])
        result = trend_provider.get_column_difference(first_column_name, second_column_name)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False)

    def test_get_current_trend(self):
        first_column_name: str = 'open'
        second_column_name: str = 'close'
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = trend_provider.get_current_trend(first_column_name, second_column_name)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False)

    def test_get_recent_trend(self):
        column_name = 'close'
        periods = 2
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
        result = trend_provider.get_recent_trend(column_name, periods)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False)


if __name__ == '__main__':
    unittest.main()
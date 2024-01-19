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
            'close': [100, 105, 110, 95, 102, 98, 105, 108, 112, 96]
        }
        self.testing_time_series: pd.DataFrame = pd.DataFrame(testing_time_series)
        self.rounding_factor: int = 4

    def test_get_absolute_change(self):
        column_name = 'close'
        periods = 2
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([np.nan, np.nan, 10.0, -10.0, -8.0, 3.0, 3.0, 10.0, 7.0, -12.0])
        result = trend_provider.get_absolute_change(column_name, periods)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=True, check_names=False)

    def test_get_percentage_change(self):
        column_name = 'close'
        periods = 2
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([np.nan, np.nan, 0.1, -0.0952, -0.0727, 0.0316, 0.0294, 0.1020, 0.0667, -0.1111])
        result = trend_provider.get_percentage_change(column_name, periods)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=True, check_names=False)

    def test_get_column_difference(self):
        first_column_name: str = 'open'
        second_column_name: str = 'close'
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([-3, 3, -7, 2, -3, 6, -4, 4, -6, 5])
        result = trend_provider.get_column_difference(first_column_name, second_column_name)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False)

    def test_get_current_trend(self):
        first_column_name: str = 'open'
        second_column_name: str = 'close'
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = trend_provider.get_current_trend(first_column_name, second_column_name)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False)

    def test_fet_recent_trend(self):
        column_name = 'close'
        periods = 2
        trend_provider: TrendProvider = TrendProvider(self.testing_time_series, self.rounding_factor)
        expected_result = pd.Series([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
        result = trend_provider.get_recent_trend(column_name, periods)
        pd.testing.assert_series_equal(result, expected_result, check_dtype=False, check_names=False)


if __name__ == '__main__':
    unittest.main()
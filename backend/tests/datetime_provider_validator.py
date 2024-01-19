import unittest
import pandas as pd
from backend.utils.datetime_provider import DatetimeProvider


class TestDatetimeProvider(unittest.TestCase):
    """
    Tests the functionality of the DatetimeProvider class. It verifies the correct conversion of datetime to day of week and hour of day respectively.
    """
    def setUp(self):
        testing_dates: dict = {'datetime': ['2024-01-15 08:30:00', '2024-01-16 12:45:00', '2024-01-17 15:00:00']}
        self.testing_dates: pd.DataFrame = pd.DataFrame(testing_dates)

    def test_get_day_series(self):
        datetime_provider = DatetimeProvider(self.testing_dates)
        expected_result = pd.Series([0, 1, 2], dtype='int32')

        self.assertTrue(expected_result.equals(datetime_provider.day_series))

    def test_get_hour_series(self):
        datetime_provider = DatetimeProvider(self.testing_dates)
        expected_result = pd.Series([8, 12, 15], dtype='int32')

        self.assertTrue(expected_result.equals(datetime_provider.hour_series))

    def test_empty_dataframe(self):
        empty_dataframe = pd.DataFrame(columns=['datetime'])
        datetime_provider = DatetimeProvider(empty_dataframe)
        expected_result = pd.Series(dtype='int32')

        self.assertTrue(expected_result.equals(datetime_provider.day_series))
        self.assertTrue(expected_result.equals(datetime_provider.hour_series))


if __name__ == '__main__':
    unittest.main()


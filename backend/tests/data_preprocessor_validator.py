import unittest
import pandas as pd
from backend.utils.data_preprocessor import DataPreprocessor


class DataPreprocessorValidator(unittest.TestCase):
    def setUp(self):

        """
        This class validates the calculations and data integrity used in DataPreprocessor. Considering performance the number of tested data rows has been capped at 500.
        :return:
        """
        self.test_data: pd.DataFrame = pd.read_csv('../data/twelvedata/feature time series (1h)/TSLA_feature_time_series.csv', nrows=500)
        self.data_preprocessor = DataPreprocessor(self.test_data)
        self.test_trend_length: int = self.data_preprocessor.trend_length

        self.test_trend_data: pd.DataFrame = self.data_preprocessor.trend_data
        self.test_target_data: pd.DataFrame = self.data_preprocessor.target_data

    def test_price_columns_existence(self):
        columns_to_check = ["open", "high", "low", "close", "volume"]

        for column_name in columns_to_check:
            self.assertIn(column_name, self.test_trend_data.columns,
                          f"Price column '{column_name}' does not exist in the DataFrame.")

    def test_calculated_previous_open(self):
        decimal_places: int = 2

        for index in range(self.test_trend_length, len(self.test_trend_data)):
            expected_previous_open: float = round(self.test_trend_data.iloc[index - self.test_trend_length]["open"], decimal_places)
            actual_previous_open: float = round(self.test_trend_data.iloc[index]["previous open"], decimal_places)
            self.assertEqual(actual_previous_open, expected_previous_open, f"Error inn Line {index}")

    def test_calculated_open_change(self):
        decimal_places: int = 2

        for index in range(self.test_trend_length, len(self.test_trend_data)):
            expected_open_change: float = round(self.test_trend_data.iloc[index]["open"] - self.test_trend_data.iloc[index]["previous open"], decimal_places)
            actual_open_change: float = self.test_trend_data.iloc[index]["open-change"]
            self.assertEqual(actual_open_change, expected_open_change, f"Error in Line {index}")

    def test_calculated_open_trend(self):
        for index in range(self.test_trend_length, len(self.test_trend_data)):

            current_open_change: float = self.test_trend_data.iloc[index]["open-change"]
            expected_open_trend: int = 1 if current_open_change > 0 else 0
            actual_open_trend: int = self.test_trend_data.iloc[index]["open-trend"]
            self.assertEqual(actual_open_trend, expected_open_trend, f"Error in Line {index}")

    def test_target_column_existence(self):
        target_column_name: str = "target"
        self.assertIn(target_column_name, self.test_target_data, f"Column {target_column_name} not in Dataframe")

    def test_time_metrics_columns_existence(self):
        time_metrics_columns = ["day_of_week", "hour"]

        for time_metric_column in time_metrics_columns:
            self.assertIn(time_metric_column, self.test_target_data.columns, f"Time metric column '{time_metric_column}' does not exist in DataFrame.")

    def test_target_data_shift(self):
        for index in range(len(self.test_target_data) - self.test_trend_length):
            expected_target: int = int(self.test_trend_data.iloc[index + self.test_trend_length + self.test_trend_length]["open-trend"])
            actual_target: int = int(self.test_target_data.iloc[index]["target"])
            self.assertEqual(actual_target, expected_target, f"Error in Line {index}")


if __name__ == '__main__':
    unittest.main()

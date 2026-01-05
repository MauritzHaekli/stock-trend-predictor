import unittest
import pandas as pd
import numpy as np
from backend.utils.data_preprocessor import DataPreprocessor


class DataPreprocessorValidator(unittest.TestCase):
    def setUp(self):

        """
        This class validates the calculations and data integrity used in the DataPreprocessor.
        :return:
        """
        self.lookback_period: int = 3
        self.target_column: str = 'close'
        self.trend_length: int = 2

    def test_get_target_data(self):
        test_data: pd.DataFrame = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', '2024-01-06'),
            'open': [103, 102, 117, 93, 105, 92],
            'close': [100, 105, 110, 95, 102, 98],
        })
        data_preprocessor = DataPreprocessor(test_data, self.lookback_period, self.target_column, self.trend_length)
        expected_result = pd.DataFrame({
            'open': [117.0, 93.0, 105.0, 92.0],
            'close': [110.0, 95.0, 102.0, 98.0],
            'target change': [10.0, -10.0, -8.0, 3.0],
            'target change(%)': [10.0, -9.5238, -7.2727, 3.1579],
            'target trend': [1, 0, 0, 1],
            'label': [0, 1, 0, 0]
        })
        expected_result.index = pd.date_range('2024-01-03', '2024-01-06')
        expected_result.index.name = 'datetime'
        expected_result.index.freq = None

        result = data_preprocessor.get_labeled_feature_data(test_data)
        self.assertEqual(result.shape, expected_result.shape)
        self.assertListEqual(result.ohlcv_columns.tolist(), expected_result.columns.tolist())
        pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)

    def test_get_feature_data(self):
        test_data: pd.DataFrame = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', '2024-01-06'),
            'open': [103, 102, 117, 93, 105, 92],
            'close': [100, 105, 110, 95, 102, 98],
            'label': [0, 1, 0, 0, 1, 0]

        })
        data_preprocessor = DataPreprocessor(test_data, self.lookback_period, self.target_column, self.trend_length)
        expected_result = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', '2024-01-06'),
            'open': [103, 102, 117, 93, 105, 92],
            'close': [100, 105, 110, 95, 102, 98]
        })
        result = data_preprocessor.get_feature_data(test_data)
        pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)

    def test_get_feature_data_batches(self):
        test_data: pd.DataFrame = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', '2024-01-08'),
            'open': [1, 2, 3, 4, 5, 6, 7, 8],
            'close': [10, 20, 30, 40, 50, 60, 70, 80]
        })
        data_preprocessor = DataPreprocessor(test_data, self.lookback_period, self.target_column, self.trend_length)
        expected_result = np.array([
            [[4, 40, 20, 100, 1], [5, 50, 20, 66.6667, 1], [6, 60, 20, 50, 1]],
            [[5, 50, 20, 66.6667, 1], [6, 60, 20, 50, 1], [7, 70, 20, 40, 1]],
            [[6, 60, 20, 50, 1], [7, 70, 20, 40, 1], [8, 80, 20, 33.3333, 1]]
        ])
        result = data_preprocessor.get_feature_data_batches(data_preprocessor.feature_data)
        self.assertTrue(np.array_equal(result, expected_result))


if __name__ == '__main__':
    unittest.main()

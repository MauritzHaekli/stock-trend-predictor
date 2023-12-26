import unittest
import pandas as pd
from utils.data_preprocessor import DataPreprocessor


class DataPreprocessorValidator(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.read_csv('../data/indicators/TSLA_indica.csv')

    def test_get_trend_data(self):
        data_preprocessor = DataPreprocessor(self.test_data)
        result = data_preprocessor.get_trend_data(self.test)

    def test_validate_open_trend(self):
        self.validator.validate_open_trend()


if __name__ == '__main__':
    unittest.main()

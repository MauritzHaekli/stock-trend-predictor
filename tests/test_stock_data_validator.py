import unittest
from stock_data_validator import StockDataValidator


class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.validator = StockDataValidator('../time_series_data/TSLA_time_series.csv')

    def test_validate_open_change(self):
        self.validator.validate_open_change()

    def test_validate_open_trend(self):
        self.validator.validate_open_trend()


if __name__ == '__main__':
    unittest.main()

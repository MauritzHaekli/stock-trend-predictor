import pandas as pd


class StockDataValidator:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def validate_open_change(self):
        # Check if open change is calculated correctly
        self.df['open-change'] = self.df['open'] - self.df['open'].shift(10)
        print(self.df['open'], self.df['open'].shift(20))
        assert all(self.df['open-change'].notnull()), "Open change contains NaN values."

    def validate_open_trend(self):
        # Check if open trend is calculated correctly
        self.df['open-trend'] = self.df['open-change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        assert all(self.df['open-trend'].notnull()), "Open trend contains NaN values."



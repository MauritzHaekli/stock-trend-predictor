import yaml
import os
import pandas as pd
from feature_provider import FeatureProvider


class FeatureDataCollector:
    def __init__(self):
        """
        This class collects and provides feature data from time series CSV files for a list of stock symbols specified in the config.yaml file.
        It processes each symbol's data using the FeatureProvider and saves the processed features to new CSV files.
        """
        with open('../config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
            self.stock_symbols = config['feature_engineering']['stock_symbols']
            self.time_series_interval = config['feature_engineering']['time_series_interval']

    def collect_and_save_features(self):

        for stock_symbol in self.stock_symbols:
            csv_file_path = f"../data/twelvedata/time series ({self.time_series_interval})/{stock_symbol}_time_series.csv"
            save_file_path = f"../data/twelvedata/feature time series ({self.time_series_interval})/{stock_symbol}_feature_time_series.csv"
            if not os.path.exists(csv_file_path):
                print(f"Warning: {csv_file_path} does not exist. Skipping {stock_symbol}.")
                continue

            try:
                time_series = pd.read_csv(csv_file_path)
                feature_provider = FeatureProvider(time_series)
                feature_time_series = feature_provider.feature_time_series
                feature_time_series.to_csv(save_file_path, index=False)
                print(f"File saved to: {save_file_path}")
            except Exception as e:
                print(f"Error processing {stock_symbol}: {e}")


if __name__ == "__main__":
    collector = FeatureDataCollector()
    collector.collect_and_save_features()

import yaml
import pandas as pd
from feature_calculator import FeatureCalculator


with open('../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

    stock_symbols: [str] = config['feature_engineering']['stock_symbols']
    time_series_interval: str = config['feature_engineering']['time_series_interval']


def generate_feature_dataframe(symbols: [str], interval: str):

    for symbol in symbols:

        csv_file_path: str = f"../data/twelvedata/time series ({interval})/{symbol}_time_series.csv"

        time_series: pd.DataFrame = pd.read_csv(csv_file_path)
        feature_calculator = FeatureCalculator(time_series)
        feature_time_series: pd.DataFrame = feature_calculator.feature_time_series

        save_file_path: str = f"../data/twelvedata/feature time series ({interval})/{symbol}_feature_time_series.csv"
        feature_time_series.to_csv(save_file_path, index=False)
        print(f"File saved to: {save_file_path}")


if __name__ == "__main__":
    generate_feature_dataframe(stock_symbols, time_series_interval)
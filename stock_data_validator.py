import pandas as pd
import yaml


class StockDataValidator:

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    def __init__(self, file_path, config_path='config.yaml'):
        self.df = pd.read_csv(file_path)
        self.config = self.load_config(config_path)
        self.trend_length = self.config['data']['trend-length']

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        return config_data

    def calculate_previous_open():
        df['previous'] = df['open'].shift(2)
        return df





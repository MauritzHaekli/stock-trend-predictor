import requests
import yaml
import pandas as pd


def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    dataframe.to_csv(output_path, index=False)
    print(f"File saved to: {output_path}")


class TimeSeriesProvider:
    def __init__(self):
        with open('../config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
            time_series_provider_parameters = config['time_series_provider_parameters']

        self.api_key = time_series_provider_parameters.get('api_key')
        self.time_series_key = time_series_provider_parameters.get('time_series_key')
        self.interval = time_series_provider_parameters.get('interval')
        self.order = time_series_provider_parameters.get('order')
        self.symbol = time_series_provider_parameters.get('symbol')
        self.output_size = time_series_provider_parameters.get('output')
        self.decimal_places = time_series_provider_parameters.get('decimal_places')
        self.dataframe_key = "datetime"
        self.response_values_key = "values"

    def get_time_series(self) -> pd.DataFrame:
        url = (f"https://api.twelvedata.com/{self.time_series_key}?"
               f"symbol={self.symbol}"
               f"&interval={self.interval}"
               f"&order={self.order}"
               f"&dp={self.decimal_places}"
               f"&outputsize={self.output_size}"
               f"&apikey={self.api_key}"
               )
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return pd.DataFrame()

        values = response.json().get(self.response_values_key, [])
        dataframe = pd.DataFrame(values)
        dataframe[self.dataframe_key] = pd.to_datetime(dataframe[self.dataframe_key])
        return dataframe

    def generate_csv(self):
        time_series_df = self.get_time_series()
        if not time_series_df.empty:
            save_file_path = f"../data/twelvedata/time series ({self.interval})/{self.symbol}_time_series.csv"
            save_to_csv(time_series_df, save_file_path)


if __name__ == "__main__":
    generator = TimeSeriesProvider()
    generator.generate_csv()

import requests
import yaml
import pandas as pd


def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """
    Saves a given DataFrame to a CSV file at the specified path.

    :param dataframe: DataFrame to be saved to CSV.
    :param output_path: Path where the CSV file will be saved.
    """
    dataframe.to_csv(output_path, index=False)
    print(f"File saved to: {output_path}")


class TimeSeriesProvider:
    def __init__(self, use_date_interval: bool = False):
        with open('C:/Users/mohae/Desktop/StockTrendPredictor/backend/config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
            params = config['time_series_provider_parameters']

        self.api_key = params.get('api_key')
        self.time_series_key = params.get('time_series_key')
        self.interval = params.get('interval')
        self.order = params.get('order')
        self.symbol = params.get('symbol')
        self.output_size = params.get('output')
        self.decimal_places = params.get('decimal_places')

        self.start_date = params.get('start_date')
        self.end_date = params.get('end_date')
        self.use_start_date = use_date_interval

        self.dataframe_key = "datetime"
        self.response_values_key = "values"

    def get_time_series(self) -> pd.DataFrame:
        url = (
            f"https://api.twelvedata.com/{self.time_series_key}?"
            f"symbol={self.symbol}"
            f"&interval={self.interval}"
            f"&order={self.order}"
            f"&dp={self.decimal_places}"
            f"&outputsize={self.output_size}"
            f"&apikey={self.api_key}"
        )

        if self.use_start_date and self.start_date and self.end_date:
            url += f"&start_date={self.start_date}"
            url += f"&end_date={self.end_date}"

        print(url)
        response = requests.get(url)
        print(response.status_code)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return pd.DataFrame()

        values = response.json().get(self.response_values_key, [])
        df = pd.DataFrame(values)

        if not df.empty:
            df[self.dataframe_key] = pd.to_datetime(df[self.dataframe_key])

        print(df)
        return df

    def generate_csv(self):
        df = self.get_time_series()
        if not df.empty:
            path = f"C:/Users/mohae/Desktop/StockTrendPredictor/backend/data/twelvedata/time series ({self.interval})/{self.symbol}/{self.symbol}_time_series.csv"
            save_to_csv(df, path)

if __name__ == "__main__":
    generator = TimeSeriesProvider(use_date_interval=False)
    generator.generate_csv()

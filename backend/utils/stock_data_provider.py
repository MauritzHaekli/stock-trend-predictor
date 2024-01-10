import requests
import json
import yaml
import pandas as pd


def generate_time_series_csv():

    with open('../config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    api_key: str = config['twelvedata']['api_key']
    time_series_key: str = config['twelvedata']['time_series_key']
    interval: str = config['twelvedata']['interval']
    order: str = config['twelvedata']['order']
    symbol: str = config['twelvedata']['symbol']
    output: float = config['twelvedata']['output']
    decimal_places = config['twelvedata']['decimal_places']

    dataframe_key: str = "datetime"
    response_values: str = "values"

    def get_time_series(indicator: str, ) -> pd.DataFrame:
        url: str = f"https://api.twelvedata.com/{indicator}?symbol={symbol}&interval={interval}&order={order}&dp={decimal_places}&outputsize={output}&apikey={api_key}"
        response = requests.get(url)
        values: json = response.json().get(response_values, [])
        dataframe: pd.DataFrame = pd.DataFrame(values)
        dataframe[dataframe_key] = pd.to_datetime(dataframe[dataframe_key])

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
        else:
            return dataframe

    time_series: pd.DataFrame = get_time_series(time_series_key)

    save_file_path: str = f"../data/twelvedata/time series ({interval})/{symbol}_time_series.csv"
    time_series.to_csv(save_file_path, index=False)
    print(f"File saved to: {save_file_path}")


if __name__ == "__main__":
    generate_time_series_csv()

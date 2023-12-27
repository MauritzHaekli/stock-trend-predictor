import requests
import json
import yaml
import pandas as pd


def generate_time_series_csv():

    with open('../config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    api_key: str = config['data']['api_key']
    interval: str = config['data']['interval']
    order: str = config['data']['order']
    symbol: str = config['data']['symbol']
    output: float = config['data']['output']
    decimal_places = config['data']['decimal_places']

    dataframe_key: str = "datetime"
    response_values: str = "values"
    time_series_indicator: str = "time_series"

    def get_response_values(indicator: str, ) -> pd.DataFrame:
        url: str = f"https://api.twelvedata.com/{indicator}?symbol={symbol}&interval={interval}&order={order}&dp={decimal_places}&outputsize={output}&apikey={api_key}"
        response = requests.get(url)
        values: json = response.json().get(response_values, [])
        dataframe: pd.DataFrame = pd.DataFrame(values)
        dataframe[dataframe_key] = pd.to_datetime(dataframe[dataframe_key])

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
        else:
            return dataframe

    time_series: pd.DataFrame = get_response_values(time_series_indicator)

    bollinger_indicator: str = "percent_b"
    macd_indicator: str = "macd"
    adx_indicator: str = "adx"
    ema_indicator: str = "ema"
    rsi_indicator: str = "rsi"

    bollinger_bands_data: pd.DataFrame = get_response_values(bollinger_indicator)
    macd_data: pd.DataFrame = get_response_values(macd_indicator)
    adx_data: pd.DataFrame = get_response_values(adx_indicator)
    ema_data: pd.DataFrame = get_response_values(ema_indicator)
    rsi_data: pd.DataFrame = get_response_values(rsi_indicator)

    indicators: pd.DataFrame = pd.merge(time_series, bollinger_bands_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, macd_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, adx_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, ema_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, rsi_data, on='datetime')

    indicators.to_csv(f"../data/indicators/{symbol}_indicators.csv", index=False)


if __name__ == "__main__":
    generate_time_series_csv()

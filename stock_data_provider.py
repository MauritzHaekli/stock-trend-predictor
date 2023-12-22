import requests
import json
import yaml
import pandas as pd


def generate_time_series_csv():

    with open('config.yaml', 'r') as config_file:
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

    trend_length: int = config['data']['trend_length']

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

    def get_trend_dataframe(original_dataframe: pd.DataFrame, column_names: [str]) -> pd.DataFrame:
        applied_trend_dataframe: pd.DataFrame = original_dataframe.copy()
        trend_increased: int = 1
        trend_decreased: int = 0

        for column_name in column_names:
            applied_comparison_column_name: str = f"previous {column_name}"
            applied_change_column_name: str = f"{column_name}-change"
            applied_trend_column_name: str = f"{column_name}-trend"
            for entry in range(0, len(original_dataframe)):
                if entry < trend_length:
                    applied_trend_dataframe.loc[entry, applied_comparison_column_name] = 0
                    applied_trend_dataframe.loc[entry, applied_change_column_name] = 0
                    applied_trend_dataframe.loc[entry, applied_trend_column_name] = trend_increased
                elif entry >= trend_length:
                    column_previous: float = round(float(applied_trend_dataframe.loc[entry - trend_length, f"{column_name}"]), decimal_places)
                    column_current: float = round(float(applied_trend_dataframe.loc[entry, f"{column_name}"]), decimal_places)
                    column_difference: float = round(column_current - column_previous, decimal_places)

                    applied_trend_dataframe.loc[entry, f"previous {column_name}"] = column_previous
                    applied_trend_dataframe.loc[entry, f"{column_name}-change"] = column_difference
                    if applied_trend_dataframe.loc[entry, f"{column_name}"] <= applied_trend_dataframe.loc[entry - trend_length, f"{column_name}"]:
                        applied_trend_dataframe.loc[entry, applied_trend_column_name] = trend_decreased
                    else:
                        applied_trend_dataframe.loc[entry, applied_trend_column_name] = trend_increased
            applied_trend_dataframe[applied_trend_column_name] = applied_trend_dataframe[applied_trend_column_name].astype(int)
        return applied_trend_dataframe

    applied_column_names: [str] = config['data']['column_names']

    trend_dataframe: pd.DataFrame = get_trend_dataframe(time_series, applied_column_names)

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

    indicators: pd.DataFrame = pd.merge(trend_dataframe, bollinger_bands_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, macd_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, adx_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, ema_data, on='datetime')
    indicators: pd.DataFrame = pd.merge(indicators, rsi_data, on='datetime')

    indicators.index = pd.to_datetime(indicators['datetime'], format='%Y-%m-%d %H:%M:%S')
    indicators.drop(['datetime'], axis=1, inplace=True)
    indicators['day_of_week'] = indicators.index.day_of_week
    indicators['hour'] = indicators.index.hour
    indicators['target'] = indicators['open-trend'].shift(-trend_length, fill_value=0).astype(int)
    indicators = indicators.iloc[trend_length:]

    indicators.to_csv(f"data/{symbol}_time_series.csv", index=False)


if __name__ == "__main__":
    generate_time_series_csv()

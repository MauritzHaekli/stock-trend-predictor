import requests
import json
import pandas as pd


def get_api_key() -> str:
    json_file_path: str = './key.json'

    with open(json_file_path, 'r') as json_file:
        json_data: json = json.load(json_file)
        api_key_property: str = "api_key"
        json_api_key: str = json_data.get(api_key_property)
    return json_api_key


def generate_time_series_csv():

    api_key: str = get_api_key()
    interval: str = "1min"
    order: str = "ASC"
    symbol: str = "TSLA"
    output: float = 5000
    decimal_places = 2

    time_series_url: str = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&order={order}&dp={decimal_places}&outputsize={output}&apikey={api_key}"
    dataframe_key: str = "datetime"
    response_values: str = "values"

    trend_length: int = 2

    def get_response_values(url: str) -> pd.DataFrame:
        response = requests.get(url)
        values: json = response.json().get(response_values, [])
        dataframe: pd.DataFrame = pd.DataFrame(values)
        dataframe[dataframe_key] = pd.to_datetime(dataframe[dataframe_key])

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
        else:
            return dataframe

    time_series: pd.DataFrame = get_response_values(time_series_url)

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

    # applied_column_names = ["open", "high", "low", "close", "volume"]
    applied_column_names = ["open"]
    trend_dataframe = get_trend_dataframe(time_series, applied_column_names)

    def get_merged_dataframes(first_dataframe: pd.DataFrame, second_dataframe: pd.DataFrame, merge_key: str) -> pd.DataFrame:
        last_datetime_first_dataframe = first_dataframe.at[0, dataframe_key]
        last_datetime_second_dataframe = first_dataframe.at[0, dataframe_key]
        merge_conflict_error_message: str = "datetime merging conflict"
        if last_datetime_first_dataframe == last_datetime_second_dataframe:
            return pd.merge(first_dataframe, second_dataframe, left_on=merge_key, right_on=merge_key)
        else:
            print(merge_conflict_error_message)

    bollinger_bands_url: str = f"https://api.twelvedata.com/percent_b?symbol={symbol}&interval={interval}&outputsize={output}&order={order}&dp={decimal_places}&apikey={api_key}"
    bollinger_bands: pd.DataFrame = get_response_values(bollinger_bands_url)

    macd_url: str = f"https://api.twelvedata.com/macd?symbol={symbol}&interval={interval}&outputsize={output}&order={order}&dp={decimal_places}&apikey={api_key}"
    macd: pd.DataFrame = get_response_values(macd_url)

    adx_url: str = f"https://api.twelvedata.com/adx?symbol={symbol}&interval={interval}&outputsize={output}&order={order}&dp={decimal_places}&apikey={api_key}"
    adx: pd.DataFrame = get_response_values(adx_url)

    ema_url: str = f"https://api.twelvedata.com/ema?symbol={symbol}&interval={interval}&outputsize={output}&order={order}&dp={decimal_places}&apikey={api_key}"
    ema: pd.DataFrame = get_response_values(ema_url)

    rsi_url: str = f"https://api.twelvedata.com/rsi?symbol={symbol}&interval={interval}&outputsize={output}&order={order}&dp={decimal_places}&apikey={api_key}"
    rsi: pd.DataFrame = get_response_values(rsi_url)

    indicators_bollinger: pd.DataFrame = get_merged_dataframes(trend_dataframe, bollinger_bands, dataframe_key)
    indicators_macd: pd.DataFrame = get_merged_dataframes(indicators_bollinger, macd, dataframe_key)
    indicators_adx: pd.DataFrame = get_merged_dataframes(indicators_macd, adx, dataframe_key)
    indicators_ema: pd.DataFrame = get_merged_dataframes(indicators_adx, ema, dataframe_key)
    indicators_rsi: pd.DataFrame = get_merged_dataframes(indicators_ema, rsi, dataframe_key)

    indicators_rsi.to_csv(f"time_series_data/{symbol}_time_series.csv", index=False)


if __name__ == "__main__":
    generate_time_series_csv()

import numpy as np
import pandas as pd
import yaml


class DataPreprocessor:
    def __init__(self, stock_time_series: pd.DataFrame):

        """
        This class is used to preprocess stock data time series in order to use it in a LSTM Neural Network. Scaling hasnt been performed yet.
        :param stock_time_series: Raw stock data containing a time series with price information and technical indicators.
        :param self.lookback_period: A period we provide for the LSTM to look back upon for each time point to make a prediction.
        :param self.target_column: The column we want to make a prediction for.
        :param self.trend_length: The period we try to predict into the future. Trend length of 10 means, we try to make a prediction for what happens in 10 time series steps.
        :param self.trend_columns: A list of columns in stock_time_series for which we want to calculate a trend (0 or 1).
        :param self.trend_data: A pandas DataFrame containing price information, technical indicators and calculated trends in self.trend_columns.
        :param self.target_data: In addition to trend_data, this pandas DataFrame contains a target column with the label we try to predict. Target data has been shifted "upwards" according to self.trend_length.
        :param self.target_data_batched: A np.array containing lists of size self.lookback_period to feed into our LSTM. Entries are still unscaled.
        :param self.target_data_batched_target: A np.array of all the labels we try to predict.
        """

        with open('../config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.lookback_period: int = config["data"]["lookback_period"]
        self.target_column: str = config["data"]["target_column"]
        self.trend_length: int = config["data"]["trend_length"]
        self.trend_columns: [str] = config["data"]["trend_columns"]

        self.stock_time_series: pd.DataFrame = stock_time_series

        self.trend_data: pd.DataFrame = self.get_trend_data(self.stock_time_series)
        self.target_data: pd.DataFrame = self.get_target_data(self.trend_data)
        self.target_data_batched: [[[float]]] = self.get_lookback_batch(self.target_data)
        self.target_data_batched_target: [float] = self.get_lookback_target(self.target_data)

    def get_trend_data(self, data: pd.DataFrame) -> pd.DataFrame:

        """
        This function takes the raw time series data of a stock and applies calculations about changes, previous data and trends of several columns like "open" and "volume" to each row.
        The "previous {column_name}" columns should contain the entries of the {column_name} a trend_length prior. So with a trend length of 10, ""previous {column_name}" should contain the
        {column_name} entry 10 rows before.
        The "{column_name}-change" columns should contain the difference of the current {column_name} entry minus the {column_name} entry a trend_length ago.
        The "{column_name}-trend" columns should contain a binary indicator (0 or 1) to indicate, if {column_name} has decreased or increased since trend_length ago
        :param data: A pandas dataframe containing stock data time series

        :return applied_trend_dataframe: A pandas dataframe in which price comparisons and trends have been added
        """
        trend_dataframe: pd.DataFrame = data.copy()

        trend_increased: int = 1
        trend_decreased: int = 0
        decimal_places: int = 2

        for column_name in self.trend_columns:
            applied_comparison_column_name: str = f"previous {column_name}"
            applied_change_column_name: str = f"{column_name}-change"
            applied_trend_column_name: str = f"{column_name}-trend"
            not_computable_placeholder: int = 0
            for entry in range(len(data)):
                if entry < self.trend_length:
                    trend_dataframe.loc[entry, applied_comparison_column_name] = not_computable_placeholder
                    trend_dataframe.loc[entry, applied_change_column_name] = not_computable_placeholder
                    trend_dataframe.loc[entry, applied_trend_column_name] = trend_increased
                elif entry >= self.trend_length:
                    column_previous: float = round(float(trend_dataframe.loc[entry - self.trend_length, f"{column_name}"]), decimal_places)
                    column_current: float = round(float(trend_dataframe.loc[entry, f"{column_name}"]), decimal_places)
                    column_difference: float = round(column_current - column_previous, decimal_places)

                    trend_dataframe.loc[entry, f"previous {column_name}"] = column_previous
                    trend_dataframe.loc[entry, f"{column_name}-change"] = column_difference
                    if trend_dataframe.loc[entry, f"{column_name}"] <= trend_dataframe.loc[entry - self.trend_length, f"{column_name}"]:
                        trend_dataframe.loc[entry, applied_trend_column_name] = trend_decreased
                    else:
                        trend_dataframe.loc[entry, applied_trend_column_name] = trend_increased
            trend_dataframe[applied_trend_column_name] = trend_dataframe[applied_trend_column_name].astype(int)
        return trend_dataframe

    def get_target_data(self, trend_data: pd.DataFrame) -> pd.DataFrame:
        target_data: pd.DataFrame = trend_data.copy()
        prediction_target_column_name: str = 'open-trend'
        columns_to_drop: [str] = ["datetime"]
        target_data.index = pd.to_datetime(target_data['datetime'], format='%Y-%m-%d %H:%M:%S')
        target_data['day_of_week'] = target_data.index.day_of_week
        target_data['hour'] = target_data.index.hour
        target_data['target'] = target_data[prediction_target_column_name].shift(-self.trend_length, fill_value=0).astype(int)
        target_data.drop(columns=columns_to_drop, axis=1, inplace=True)
        target_data = target_data.iloc[self.trend_length:]
        return target_data

    def get_lookback_batch(self, data: pd.DataFrame) -> [[[float]]]:
        time_series_batch = []
        for row in range(len(data) - self.lookback_period):
            time_series_batch.append(data.iloc[row:row + self.lookback_period].values)

        time_series_batch: [[[float]]] = np.array(time_series_batch)
        return time_series_batch

    def get_lookback_target(self, data: pd.DataFrame) -> [float]:
        time_series_target = []
        for row in range(len(data) - self.lookback_period):
            time_series_target.append(data[self.target_column].iloc[row + self.lookback_period - 1])

        time_series_target: [float] = np.array(time_series_target)
        return time_series_target


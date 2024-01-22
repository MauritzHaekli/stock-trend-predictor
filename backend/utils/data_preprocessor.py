import numpy as np
import pandas as pd
from backend.utils.trend_provider import TrendProvider


class DataPreprocessor:
    def __init__(self, time_series: pd.DataFrame, lookback_period: int, target: str, trend_length: int):

        """
        This class is used to preprocess time series data in order to use it in a LSTM Neural Network. Scaling hasnt been performed yet.
        :param time_series: Raw stock data containing a time series with OHLC time series data and derived features like technical indicators and price trends.
        :param self.lookback_period: A period we provide for the LSTM to look back upon for each time point to make a prediction.
        :param self.target_column: The column we want to make a prediction for.
        :param self.trend_length: The period we try to predict into the future. Trend length of 10 means, we try to make a prediction for what happens in 10 time series steps.
        :param self.trend_data: A pandas DataFrame containing price information, technical indicators and calculated trends in self.trend_columns.
        :param self.target_data: In addition to trend_data, this pandas DataFrame contains a target column with the label we try to predict. Target data has been shifted "upwards" according to self.trend_length.
        :param self.target_data_batched: A np.array containing lists of size self.lookback_period to feed into our LSTM. Entries are still unscaled.
        :param self.target_data_batched_target: A np.array of all the labels we try to predict.
        """
        self.lookback_period: int = lookback_period
        self.target: str = target
        self.trend_length: int = trend_length

        self.label_column: str = "label"
        self.target_absolute_change_column: str = "target change"
        self.target_percentage_change_column: str = "target change(%)"
        self.target_trend_column: str = "target trend"

        self.time_series: pd.DataFrame = time_series
        self.target_data: pd.DataFrame = self.get_target_data(self.time_series)
        self.feature_data: pd.DataFrame = self.get_feature_data(self.target_data)
        self.feature_data_batched: [[[float]]] = self.get_feature_data_batches(self.feature_data)
        self.label_data: [float] = self.get_lookback_labels(self.target_data)

    def get_target_data(self, time_series: pd.DataFrame) -> pd.DataFrame:
        target_data: pd.DataFrame = time_series.copy()
        columns_to_drop: [str] = ["datetime"]
        target_rounding_factor: int = 4
        target_trend_provider: TrendProvider = TrendProvider(time_series=target_data, rounding_factor=target_rounding_factor)
        target_data.index = pd.to_datetime(target_data['datetime'], format='%Y-%m-%d %H:%M:%S')
        target_data[self.target_absolute_change_column]: pd.Series = target_trend_provider.get_absolute_change(self.target, periods=self.trend_length)
        target_data[self.target_percentage_change_column]: pd.Series = target_trend_provider.get_percentage_change(self.target, periods=self.trend_length)
        target_data[self.target_trend_column]: pd.Series = target_trend_provider.get_recent_trend(self.target, periods=self.trend_length)
        target_data[self.label_column]: pd.Series = target_data[self.target_trend_column].shift(-self.trend_length, fill_value=0)
        target_data.drop(columns=columns_to_drop, axis=1, inplace=True)
        target_data = target_data.iloc[self.trend_length:]
        return target_data

    def get_feature_data(self, target_data: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop: [str] = [self.label_column]
        feature_data: pd.DataFrame = target_data.drop(columns=columns_to_drop, axis=1)
        return feature_data

    def get_feature_data_batches(self, data: pd.DataFrame) -> [[[float]]]:
        time_series_batch: [[[float]]] = []
        feature_df: pd.DataFrame = data
        for row in range(len(feature_df) - self.lookback_period):
            time_series_batch.append(feature_df.iloc[row + 1:row + self.lookback_period + 1].values)

        time_series_batch = np.array(time_series_batch)
        return time_series_batch

    def get_lookback_labels(self, data: pd.DataFrame) -> [float]:
        time_series_target = []
        for row in range(len(data) - self.lookback_period):
            time_series_target.append(data[self.label_column].iloc[row + self.lookback_period])

        time_series_target: [float] = np.array(time_series_target)
        return time_series_target


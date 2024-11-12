import pandas as pd
import numpy as np


class TrendProvider:
    def __init__(self, time_series: pd.Series, rounding_factor: int):
        self.time_series: pd.Series = time_series
        self.rounding_factor: int = rounding_factor

    def get_absolute_change(self, periods: int) -> pd.Series:
        """
        Calculate the absolute change of a time series column for a period with length of periods.

        :param column_name: Name of the time series column we want to calculate an absolute period change for.
        :param periods: The length of the absolute change period we want to calculate. A period of 1 calculates the absolute change from t-1 to t.
        :return: A pandas Series of rounded absolute change data.
        """
        absolute_change: pd.Series = self.time_series.diff(periods=periods)
        absolute_change_rounded: pd.Series = absolute_change.round(self.rounding_factor)
        return absolute_change_rounded

    def get_percentage_change(self, periods: int) -> pd.Series:
        """
        Calculate the percentage change of a time series column for a period with length of periods.

        :param column_name: Name of the time series column we want to calculate a percentage period change for.
        :param periods: The length of the percentage change period we want to calculate. A period of 1 calculates the percentage change from t-1 to t.
        :return: A pandas Series of rounded percentage change data.
        """
        percentage_change: pd.Series = self.time_series.pct_change(periods=periods).replace([np.inf, -np.inf], 1.0000) * 100
        percentage_change_rounded: pd.Series = percentage_change.round(self.rounding_factor)
        return percentage_change_rounded

    def get_recent_trend(self, periods: int) -> pd.Series:
        """
        Calculates the trend for a time series column for a trend period. With a trend period of 1 calculates the trend of a price from t-1 to t.
        :param column_name: The name of the column we want to calculate a trend for.
        :param periods: The length of periods we want to look back upon to calculate a trend.
        :return: A pandas Series containing recent price trends.
        """
        recent_trend: pd.Series = (self.time_series > self.time_series.shift(periods)).astype(int)
        return recent_trend




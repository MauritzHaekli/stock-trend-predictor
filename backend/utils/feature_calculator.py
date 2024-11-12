import pandas as pd
import numpy as np


class FeatureCalculator:
    def __init__(self, feature: pd.Series, periods: int, rounding_factor: int):
        self.feature: pd.Series = feature
        self.feature_name = self.feature.name
        self.rounding_factor: int = rounding_factor
        self.latest_period: int = 1
        self.recent_period: int = periods

        self.latest_absolute_change: pd.Series = self.get_absolute_change(self.latest_period)
        self.recent_absolute_change: pd.Series = self.get_absolute_change(self.recent_period)

        self.latest_percentage_change: pd.Series = self.get_percentage_change(self.latest_period)
        self.recent_percentage_change: pd.Series = self.get_percentage_change(self.recent_period)

        self.latest_trend_change: pd.Series = self.get_trend_change(self.latest_period)
        self.recent_trend_change: pd.Series = self.get_trend_change(self.recent_period)

    def get_absolute_change(self, periods: int) -> pd.Series:
        """
        Calculate the absolute change of a time series column for a period with length of periods.
        :param periods: The length of the absolute change period we want to calculate. A period of 1 calculates the absolute change from t-1 to t.
        :return: A pandas Series of rounded absolute change data.
        """
        absolute_change_name: str = f"{self.feature_name}_{periods} change"
        absolute_change: pd.Series = self.feature.diff(periods=periods)
        absolute_change_rounded: pd.Series = pd.Series(absolute_change.round(self.rounding_factor), name=absolute_change_name)
        return absolute_change_rounded

    def get_percentage_change(self, periods: int) -> pd.Series:
        """
        Calculate the percentage change of a time series column for a period with length of periods.
        :return: A pandas Series of rounded percentage change data.
        """
        percentage_change_name: str = f"{self.feature_name}_{periods} change(%)"
        percentage_change: pd.Series = self.feature.pct_change(periods=periods).replace([np.inf, -np.inf], 1.0000) * 100
        percentage_change_rounded: pd.Series = pd.Series(percentage_change.round(self.rounding_factor), name=percentage_change_name)
        return percentage_change_rounded

    def get_trend_change(self, periods: int) -> pd.Series:
        recent_trend_name: str = f"{self.feature_name}_{periods} trend"
        trend: pd.Series = pd.Series((self.feature > self.feature.shift(periods)).astype(int), name=recent_trend_name)
        return trend



import pandas as pd
import numpy as np


class FeatureChangeCalculator:
    """
        Calculates various feature transformations for a given time series feature column.

        This class computes:
            - Absolute changes over specified periods.
            - Percentage changes over specified periods.
            - Trend changes (1 for positive trend, 0 for no change or negative trend).

        Attributes:
            feature (pd.Series): The input time series data.
            periods (int): The period length for calculating changes.
            rounding_factor (int): The number of decimal places to round results to.
        """
    def __init__(self, feature: pd.Series, periods: int, rounding_factor: int):
        """
        Initializes the FeatureCalculator with the feature data, period, and rounding factor.

        :param feature: A pandas Series representing the time series data.
        :param periods: Number of periods to use for calculating recent changes.
        :param rounding_factor: Number of decimal places to round the calculated values.
        """
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
        Calculate the absolute change of the feature over the specified number of periods.

        :param periods: The period over which to calculate absolute change. A period of `n`
                        calculates the change between time `t` and `t-n`.
        :return: A pandas Series containing the rounded absolute change values.
        """
        absolute_change_name: str = f"{self.feature_name}_{periods} change"
        absolute_change: pd.Series = self.feature.diff(periods=periods)
        absolute_change_rounded: pd.Series = pd.Series(absolute_change.round(self.rounding_factor), name=absolute_change_name)
        return absolute_change_rounded

    def get_percentage_change(self, periods: int) -> pd.Series:
        """
        Calculate the percentage change of the feature over the specified number of periods.

        :param periods: The period over which to calculate percentage change.
        :return: A pandas Series containing the rounded percentage change values.
        """
        percentage_change_name: str = f"{self.feature_name}_{periods} change(%)"
        percentage_change: pd.Series = self.feature.pct_change(periods=periods).replace([np.inf, -np.inf], 1.0000) * 100
        percentage_change_rounded: pd.Series = pd.Series(percentage_change.round(self.rounding_factor), name=percentage_change_name)
        return percentage_change_rounded

    def get_trend_change(self, periods: int) -> pd.Series:
        """
        Calculate a binary trend change indicator over the specified number of periods.

        The trend is positive (1) if the current value is greater than the value `n` periods ago;
        otherwise, it is non-positive (0).

        :param periods: The period over which to determine the trend.
        :return: A pandas Series containing the trend indicators (1 for positive trend, 0 otherwise).
        """
        recent_trend_name: str = f"{self.feature_name}_{periods} trend"
        trend: pd.Series = pd.Series((self.feature > self.feature.shift(periods)).astype(int), name=recent_trend_name)
        return trend

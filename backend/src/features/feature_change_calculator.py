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
            rounding_factor (int): The number of decimal places to round results to.
        """
    def __init__(self, rounding_factor: int = 2):
        """
        Initializes the FeatureCalculator with the feature data, and rounding factor.

        :param rounding_factor: Number of decimal places to round the calculated values.
        """
        self.rounding_factor: int = rounding_factor

    def get_absolute_change(self, feature: pd.Series, periods: int) -> pd.Series:
        """
        Calculate the absolute change of the feature over the specified number of periods.

        :return: A pandas Series containing the rounded absolute change values.
        """
        absolute_change_name: str = f"{feature.name}({periods}) change"
        absolute_change: pd.Series = feature.diff(periods=periods)
        absolute_change_rounded: pd.Series = pd.Series(absolute_change.round(self.rounding_factor), name=absolute_change_name)
        return absolute_change_rounded

    def get_relative_change(self, feature: pd.Series, periods: int, inf_replacement: float = np.nan) -> pd.Series:
        """
        Calculate the percentage change of the feature over the specified number of periods.

        :return: A pandas Series containing the rounded percentage change values.
        """

        percentage_change_name: str = f"{feature.name}({periods}) change(%)"
        percentage_change: pd.Series = feature.pct_change(periods=periods) * 100
        percentage_change = percentage_change.replace([np.inf, -np.inf], inf_replacement)
        percentage_change_rounded: pd.Series = pd.Series(percentage_change.round(self.rounding_factor),
                                                         name=percentage_change_name)
        return percentage_change_rounded

    @staticmethod
    def get_trend_change(feature: pd.Series, periods: int) -> pd.Series:
        """
        Calculate a binary trend change indicator over the specified number of periods.

        The trend is positive (1) if the current value is greater than the value `n` periods ago;
        otherwise, it is non-positive (0).

        :return: A pandas Series containing the trend indicators (1 for positive trend, 0 otherwise).
        """

        recent_trend_name: str = f"{feature.name}({periods}) trend"
        trend_change: pd.Series = pd.Series(
            np.where(feature > feature.shift(periods), 1, 0),
            name=recent_trend_name
        )
        return trend_change

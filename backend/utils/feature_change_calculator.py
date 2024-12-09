import pandas as pd
import numpy as np
from backend.utils.validator_service import BaseValidator, PeriodValidator


class FeatureChangeCalculator:
    """
        Calculates various feature transformations for a given time series feature column.

        This class computes:
            - Absolute changes over specified periods.
            - Percentage changes over specified periods.
            - Trend changes (1 for positive trend, 0 for no change or negative trend).

        Attributes:
            feature (pd.Series): The input time series data.
            rounding_factor (int): The number of decimal places to round results to.
        """
    def __init__(self, feature: pd.Series, rounding_factor: int = 2):
        """
        Initializes the FeatureCalculator with the feature data, and rounding factor.

        :param feature: A pandas Series representing the time series data.
        :param rounding_factor: Number of decimal places to round the calculated values.
        """
        self.feature: pd.Series = feature
        self.feature_name = self.feature.name
        self.rounding_factor: int = rounding_factor
        self.base_validator = BaseValidator
        self.period_validator = PeriodValidator
        self.base_validator.validate_series(self.feature)

    def get_absolute_change(self, periods: int) -> pd.Series:
        """
        Calculate the absolute change of the feature over the specified number of periods.

        :param periods: The period over which to calculate absolute change. A period of `n`
                        calculates the change between time `t` and `t-n`.
        :return: A pandas Series containing the rounded absolute change values.
        """
        absolute_change_name: str = f"{self.feature_name}({periods}) change"
        absolute_change: pd.Series = self.feature.diff(periods=periods)
        absolute_change_rounded: pd.Series = pd.Series(absolute_change.round(self.rounding_factor), name=absolute_change_name)
        return absolute_change_rounded

    def get_relative_change(self, periods: int, inf_replacement: float = np.nan) -> pd.Series:
        """
        Calculate the percentage change of the feature over the specified number of periods.

        :param periods: The period over which to calculate percentage change.
        :param inf_replacement: Value to replace infinities with. Default is NaN.
        :return: A pandas Series containing the rounded percentage change values.
        """
        self.period_validator.validate_periods(periods, self.feature)

        percentage_change_name: str = f"{self.feature_name}({periods}) change(%)"
        percentage_change: pd.Series = self.feature.pct_change(periods=periods) * 100
        percentage_change = percentage_change.replace([np.inf, -np.inf], inf_replacement)
        percentage_change_rounded: pd.Series = pd.Series(percentage_change.round(self.rounding_factor),
                                                         name=percentage_change_name)
        return percentage_change_rounded

    def get_trend_change(self, periods: int) -> pd.Series:
        """
        Calculate a binary trend change indicator over the specified number of periods.

        The trend is positive (1) if the current value is greater than the value `n` periods ago;
        otherwise, it is non-positive (0).

        :param periods: The period over which to determine the trend.
        :return: A pandas Series containing the trend indicators (1 for positive trend, 0 otherwise).
        """
        self.period_validator.validate_periods(periods, self.feature)

        recent_trend_name: str = f"{self.feature_name}({periods}) trend"
        trend_change: pd.Series = pd.Series(
            np.where(self.feature > self.feature.shift(periods), 1, 0),
            name=recent_trend_name
        )
        return trend_change

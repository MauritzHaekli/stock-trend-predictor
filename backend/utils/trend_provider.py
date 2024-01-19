import pandas as pd
from backend.utils.feature_column_names import FeatureColumnNames


class TrendProvider:
    def __init__(self, time_series: pd.DataFrame, rounding_factor: int):
        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.time_series: pd.DataFrame = time_series
        self.rounding_factor: int = rounding_factor

    def get_absolute_change(self, column_name: str, periods: int) -> pd.Series:
        """
        Calculate the absolute change of a time series column for a period with length of periods.

        :param column_name: Name of the time series column we want to calculate an absolute period change for.
        :param periods: The length of the absolute change period we want to calculate. A period of 1 calculates the absolute change from t-1 to t.
        :return: A pandas Series of rounded absolute change data.
        """
        absolute_change_column: pd.Series = self.time_series[column_name]
        absolute_change: pd.Series = absolute_change_column.diff(periods=periods)
        absolute_change_rounded: pd.Series = absolute_change.round(self.rounding_factor)
        return absolute_change_rounded

    def get_percentage_change(self, column_name: str, periods: int) -> pd.Series:
        """
        Calculate the percentage change of a time series column for a period with length of periods.

        :param column_name: Name of the time series column we want to calculate a percentage period change for.
        :param periods: The length of the percentage change period we want to calculate. A period of 1 calculates the percentage change from t-1 to t.
        :return: A pandas Series of rounded percentage change data.
        """
        percentage_change_column: pd.Series = self.time_series[column_name]
        percentage_change: pd.Series = percentage_change_column.pct_change(periods=periods)
        percentage_change_rounded: pd.Series = percentage_change.round(self.rounding_factor)
        return percentage_change_rounded

    def get_recent_trend(self, trend_column: str, periods: int) -> pd.Series:
        """
        Calculates the trend for a time series column for a trend period. With a trend period of 1 calculates the trend of a price from t-1 to t.
        :param trend_column: The name of the column we want to calculate a trend for.
        :param periods: The length of periods we want to look back upon to calculate a trend.
        :return: A pandas Series containing recent price trends.
        """
        recent_trend: pd.Series = (self.time_series[trend_column] > self.time_series[trend_column].shift(periods)).astype(int)
        return recent_trend

    def get_column_difference(self, starting_column: str, final_column: str) -> pd.Series:
        """
        Calculates the difference between two time series columns.

        :param starting_column: The starting column we want to calculate a difference for. Typically used for either the opening or high price.
        :param final_column: The final column we want to calculate a difference for. Typically used for either the close or low price.
        :return: A pandas Series containing the rounded difference of two time series columns.
        """
        start_column: pd.Series = self.time_series[starting_column]
        final_column: pd.Series = self.time_series[final_column]
        column_difference: pd.Series = final_column - start_column
        column_difference_rounded: pd.Series = column_difference.round(self.rounding_factor)
        return column_difference_rounded

    def get_current_trend(self, first_column, second_column) -> pd.Series:
        """
        Calculate the current trend of a time series row, usually used the trend of opening price to closing price. A trend is either increased values (1) or decreased values (0) since the last period.
        :return: A pandas Series containing column trends.
        """
        current_column_differences: pd.Series = self.get_column_difference(first_column, second_column)
        current_column_trend: pd.Series = (current_column_differences > 0).astype(int)
        return current_column_trend


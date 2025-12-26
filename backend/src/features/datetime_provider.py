import pandas as pd


class DatetimeProvider:
    def __init__(self, time_series: pd.DataFrame, datetime_column: str = None):
        """
        Calculate the day of week and hour of day from a time series containing a datetime column.

        :param time_series: A time series with a column containing datetime information.
        :param datetime_column: The name of the datetime column. Defaults to `FeatureColumnNames.DATETIME`.
        """
        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")

        self.index = time_series.index

    def get_day_series(self) -> pd.Series:
        """
        Extracts the day of the week from the datetime column.

        :return: A pandas Series with integer values representing the day of the week (0 = Monday, 6 = Sunday).
        """
        return pd.Series(self.index.hour, index=self.index, name="hour")

    def get_hour_series(self) -> pd.Series:
        """
        Extracts the hour of the day from the datetime column.

        :return: A pandas Series with integer values representing the hour of the day.
        """
        return self.datetime_series.dt.hour

    def get_day_and_hour_series(self) -> pd.DataFrame:
        """
        Extracts both the day of the week and the hour of the day as a DataFrame.

        :return: A pandas DataFrame with two columns: 'day_of_week' and 'hour_of_day'.
        """
        return pd.DataFrame({
            "day_of_week": self.datetime_series.dt.dayofweek,
            "hour_of_day": self.datetime_series.dt.hour
        })

import pandas as pd
from backend.utils.feature_column_names import FeatureColumnNames


class DatetimeProvider:
    def __init__(self, time_series: pd.DataFrame):
        """
        Calculate the day of week and hour of day from a time series containing a datetime column.
        :param time_series: A time series with a column containing datetime information.
        """
        self.time_series: pd.DataFrame = time_series
        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.day_series: pd.Series = self.get_day_series(self.time_series)
        self.hour_series: pd.Series = self.get_hour_series(self.time_series)

    def get_day_series(self, time_series: pd.DataFrame) -> pd.Series:
        datetime_series: pd.Series = pd.to_datetime(time_series[self.column_names.datetime])
        day_series: pd.Series = datetime_series.dt.dayofweek
        return day_series

    def get_hour_series(self, time_series: pd.DataFrame) -> pd.Series:
        datetime_series: pd.Series = pd.to_datetime(time_series[self.column_names.datetime])
        hour_series: pd.Series = datetime_series.dt.hour
        return hour_series

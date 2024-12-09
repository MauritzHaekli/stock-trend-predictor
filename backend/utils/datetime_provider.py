import pandas as pd
from backend.utils.feature_column_names import FeatureColumnNames
from backend.utils.validator_service import DatetimeValidator


class DatetimeProvider:
    def __init__(self, time_series: pd.DataFrame, datetime_column: str = None, validator: DatetimeValidator = None):
        """
        Calculate the day of week and hour of day from a time series containing a datetime column.

        :param time_series: A time series with a column containing datetime information.
        :param datetime_column: The name of the datetime column. Defaults to `FeatureColumnNames.DATETIME`.
        :param validator: An instance of a DatetimeValidator to validate the time series.
        """
        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.datetime_column = datetime_column or self.column_names.DATETIME
        self.validator = validator or DatetimeValidator()

        # Validate the DataFrame
        self.validator.validate_datetime_feature(time_series)

        try:
            self.datetime_series: pd.Series = (
                time_series[self.datetime_column]
                if pd.api.types.is_datetime64_any_dtype(time_series[self.datetime_column])
                else pd.to_datetime(time_series[self.datetime_column])
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse datetime column '{self.datetime_column}' in the DataFrame: {e}"
            )

    def get_day_series(self) -> pd.Series:
        """
        Extracts the day of the week from the datetime column.

        :return: A pandas Series with integer values representing the day of the week (0 = Monday, 6 = Sunday).
        """
        return self.datetime_series.dt.dayofweek

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

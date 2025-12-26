import pandas as pd


class DatetimeProvider:
    """
    Extracts time-based features from a DataFrame with a DatetimeIndex.

    Assumes:
    - Datetime information is stored in the DataFrame index
    - Index is timezone-consistent (or already normalized)
    """

    def __init__(self, time_series: pd.DataFrame):
        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a pandas DatetimeIndex")

        self.index: pd.DatetimeIndex = time_series.index

    def day_of_week(self) -> pd.Series:
        """
        Day of week (0 = Monday, 6 = Sunday)
        """
        return pd.Series(
            self.index.dayofweek,
            index=self.index,
            name="day_of_week",
        )

    def hour_of_day(self) -> pd.Series:
        """
        Hour of day (0–23)
        """
        return pd.Series(
            self.index.hour,
            index=self.index,
            name="hour_of_day",
        )

    def day_and_hour(self) -> pd.DataFrame:
        """
        Day of week and hour of day as a DataFrame.
        """
        return pd.DataFrame(
            {
                "day_of_week": self.index.dayofweek,
                "hour_of_day": self.index.hour,
            },
            index=self.index,
        )

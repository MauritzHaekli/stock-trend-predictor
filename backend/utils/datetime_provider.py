import pandas as pd


class DatetimeProvider:
    def __init__(self, time_series: pd.DataFrame):
        self.time_series: pd.DataFrame = time_series

    def get_
import pandas as pd
import numpy as np
from backend.src.models.feature_column_names import FeatureColumnNames


class PriceFeatureProvider:

    def __init__(self, time_series: pd.DataFrame):
        self.time_series: pd.DataFrame = time_series
        self.feature_column_names = FeatureColumnNames
        self.price_return: pd.Series = self.get_price_return()
        self.log_return: pd.Series = self.get_log_return()
        self.price_volatility: pd.Series = self.get_price_volatility()
        self.price_difference: pd.Series = self.get_price_difference()

    def get_price_return(self) -> pd.Series:
        price_return: pd.Series = ((self.time_series[self.feature_column_names.CLOSE_PRICE] - self.time_series[self.feature_column_names.OPEN_PRICE])/self.time_series[self.feature_column_names.OPEN_PRICE]) * 100
        price_return_rounded: pd.Series = price_return.round(3)
        return price_return_rounded

    def get_log_return(self) -> pd.Series:
        log_return: pd.Series = np.log(self.time_series[self.feature_column_names.CLOSE_PRICE]/self.time_series[self.feature_column_names.CLOSE_PRICE].shift(1)) * 100
        log_return_rounded: pd.Series = log_return.round(3)
        return log_return_rounded

    def get_price_volatility(self) -> pd.Series:
        price_volatility: pd.Series = self.time_series[self.feature_column_names.HIGH_PRICE] - self.time_series[self.feature_column_names.LOW_PRICE]
        return price_volatility

    def get_price_difference(self) -> pd.Series:
        price_difference: pd.Series = self.time_series[self.feature_column_names.CLOSE_PRICE] - self.time_series[self.feature_column_names.OPEN_PRICE]
        return price_difference


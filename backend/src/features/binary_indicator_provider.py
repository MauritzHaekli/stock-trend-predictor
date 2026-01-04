import pandas as pd
from backend.src.schema.binary_indicators import BinaryIndicatorColumns


class BinaryIndicatorProvider:
    """
    Computes geometric and dynamic relationships between price and core indicators
    (e.g. EMA distance, slope, acceleration).
    """

    def __init__(
        self,
        close_price: pd.Series,
        ema_series: pd.Series,
        ema_slope: pd.Series,
        log_return: pd.Series,
        atr_series: pd.Series,
        bollinger_series: pd.Series,
    ):
        self.close = close_price
        self.ema = ema_series
        self.ema_slope = ema_slope
        self.log_return = log_return
        self.atr_series = atr_series
        self.bollinger_series = bollinger_series

        # 2. Align indices (DROP rows not common to all)
        self.close, self.ema = self.close.align(self.ema, join="inner")
        self.close, self.ema_slope = self.close.align(self.ema_slope, join="inner")
        self.close, self.log_return = self.close.align(self.log_return, join="inner")
        self.close, self.atr_series = self.close.align(self.atr_series, join="inner")
        self.close, self.bollinger_series = self.close.align(self.bollinger_series, join="inner")

        self.binary_indicator_columns = BinaryIndicatorColumns()

    @property
    def binary_indicators(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.binary_indicator_columns.CLOSE_ABOVE_EMA: self.close_above_ema,
                self.binary_indicator_columns.EMA_SLOPE_POSITIVE: self.ema_slope_positive,
                self.binary_indicator_columns.LOG_RETURN_POSITIVE: self.log_return_positive,
                self.binary_indicator_columns.ATR_STRETCH: self.atr_stretch,
                self.binary_indicator_columns.BOLLINGER_ABOVE_ONE: self.bollinger_above_one,
                self.binary_indicator_columns.BOLLINGER_UNDER_ZERO: self.bollinger_under_zero,
            },
            index=self.close.index,
        )


    @property
    def close_above_ema(self) -> pd.Series:
        close_above_ema = (self.close > self.ema).astype(int)
        close_above_ema.name = self.binary_indicator_columns.CLOSE_ABOVE_EMA
        return close_above_ema

    @property
    def ema_slope_positive(self) -> pd.Series:
        ema_slope_positive = (self.ema_slope > 0).astype(int)
        ema_slope_positive.name = self.binary_indicator_columns.EMA_SLOPE_POSITIVE
        return ema_slope_positive

    @property
    def log_return_positive(self) -> pd.Series:
        log_return_positive = (self.log_return > 0).astype(int)
        log_return_positive.name = self.binary_indicator_columns.LOG_RETURN_POSITIVE
        return log_return_positive

    @property
    def atr_stretch(self) -> pd.Series:
        atr_stretch = (abs(self.close - self.ema) > self.atr_series).astype(int)
        atr_stretch.name = self.binary_indicator_columns.ATR_STRETCH
        return atr_stretch

    @property
    def bollinger_above_one(self) -> pd.Series:
        bollinger_above_one = (self.bollinger_series > 1).astype(int)
        bollinger_above_one.name = self.binary_indicator_columns.BOLLINGER_ABOVE_ONE
        return bollinger_above_one

    @property
    def bollinger_under_zero(self) -> pd.Series:
        bollinger_under_zero = (self.bollinger_series < 0).astype(int)
        bollinger_under_zero.name = self.binary_indicator_columns.BOLLINGER_UNDER_ZERO
        return bollinger_under_zero

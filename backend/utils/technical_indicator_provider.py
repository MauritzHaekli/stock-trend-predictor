import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volume import VolumePriceTrendIndicator
from backend.utils.feature_column_names import FeatureColumnNames


class TechnicalIndicatorProvider:
    def __init__(self, time_series: pd.DataFrame, rounding_factor: int):
        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.rounding_factor: int = rounding_factor
        self.time_series: pd.DataFrame = time_series
        self.bollinger_period: int = 20
        self.bollinger_std: int = 2
        self.atr_window: int = 14
        self.macd_short_period: int = 12
        self.macd_long_period: int = 26
        self.macd_signal_period: int = 9
        self.adx_period: int = 14
        self.sma_period: int = 9
        self.ema_period: int = 9
        self.rsi_period: int = 14
        self.stoch_window: int = 14
        self.stoch_smooth_window: int = 3

        self.sma: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.close_price],
                                              window=self.sma_period,
                                              fillna=True)
        self.ema: EMAIndicator = EMAIndicator(close=self.time_series[self.column_names.close_price],
                                              window=self.ema_period,
                                              fillna=True)

        self.bb: BollingerBands = BollingerBands(close=self.time_series[self.column_names.close_price],
                                                 window=self.bollinger_period,
                                                 window_dev=self.bollinger_std,
                                                 fillna=True)
        self.atr: AverageTrueRange = AverageTrueRange(high=self.time_series[self.column_names.high_price],
                                                      low=self.time_series[self.column_names.low_price],
                                                      close=self.time_series[self.column_names.close_price],
                                                      window=self.atr_window,
                                                      fillna=True)
        self.macd: MACD = MACD(close=self.time_series[self.column_names.close_price],
                               window_fast=self.macd_short_period,
                               window_slow=self.macd_long_period,
                               window_sign=self.macd_signal_period,
                               fillna=True)
        self.adx: ADXIndicator = ADXIndicator(high=self.time_series[self.column_names.high_price],
                                              low=self.time_series[self.column_names.low_price],
                                              close=self.time_series[self.column_names.close_price],
                                              window=self.adx_period,
                                              fillna=True)

        self.rsi: RSIIndicator = RSIIndicator(self.time_series[self.column_names.close_price],
                                              window=self.rsi_period,
                                              fillna=True)
        self.stoch: StochasticOscillator = StochasticOscillator(high=self.time_series[self.column_names.high_price],
                                                                low=self.time_series[self.column_names.low_price],
                                                                close=self.time_series[self.column_names.close_price],
                                                                window=self.stoch_window,
                                                                smooth_window=self.stoch_smooth_window,
                                                                fillna=True)
        self.vpt: VolumePriceTrendIndicator = VolumePriceTrendIndicator(close=self.time_series[self.column_names.close_price],
                                                                        volume=self.time_series[self.column_names.volume],
                                                                        fillna=True)
        self.technical_indicators: pd.DataFrame = self.get_technical_indicators()

    def get_technical_indicators(self) -> pd.DataFrame:
        technical_indicators: pd.DataFrame = pd.DataFrame()
        technical_indicators[self.column_names.sma_price]: pd.Series = self.sma.sma_indicator().round(self.rounding_factor)
        technical_indicators[self.column_names.ema_price]: pd.Series = self.ema.ema_indicator().round(self.rounding_factor)
        technical_indicators[self.column_names.percent_b]: pd.Series = self.bb.bollinger_pband().round(self.rounding_factor)
        technical_indicators[self.column_names.atr]: pd.Series = self.atr.average_true_range().round(self.rounding_factor)
        technical_indicators[self.column_names.macd]: pd.Series = self.macd.macd().round(self.rounding_factor)
        technical_indicators[self.column_names.macd_signal]: pd.Series = self.macd.macd_signal().round(self.rounding_factor)
        technical_indicators[self.column_names.macd_hist]: pd.Series = self.macd.macd_diff().round(self.rounding_factor)
        technical_indicators[self.column_names.adx]: pd.Series = self.adx.adx().round(self.rounding_factor)
        technical_indicators[self.column_names.rsi]: pd.Series = self.rsi.rsi().round(self.rounding_factor)
        technical_indicators[self.column_names.fast_stochastic]: pd.Series = self.stoch.stoch().round(self.rounding_factor)
        technical_indicators[self.column_names.slow_stochastic]: pd.Series = self.stoch.stoch_signal().round(self.rounding_factor)
        technical_indicators[self.column_names.vpt]: pd.Series = self.vpt.volume_price_trend().round(self.rounding_factor)
        return technical_indicators

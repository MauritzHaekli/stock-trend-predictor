import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volume import VolumePriceTrendIndicator
from feature_column_names import FeatureColumnNames


class TechnicalIndicatorProvider:
    def __init__(self, time_series: pd.DataFrame):
        self.column_names = FeatureColumnNames()
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

        self.bb: BollingerBands = BollingerBands(close=time_series[self.column_names.close_price],
                                                 window=self.bollinger_period,
                                                 window_dev=self.bollinger_std,
                                                 fillna=True)
        self.atr: AverageTrueRange = AverageTrueRange(high=time_series[self.column_names.high_price],
                                                      low=time_series[self.column_names.low_price],
                                                      close=time_series[self.column_names.close_price],
                                                      window=self.atr_window,
                                                      fillna=True)
        self.macd: MACD = MACD(close=time_series[self.column_names.close_price],
                               window_fast=self.macd_short_period,
                               window_slow=self.macd_long_period,
                               window_sign=self.macd_signal_period,
                               fillna=True)
        self.adx: ADXIndicator = ADXIndicator(high=time_series[self.column_names.high_price],
                                              low=time_series[self.column_names.low_price],
                                              close=time_series[self.column_names.close_price],
                                              window=self.adx_period,
                                              fillna=True)
        self.sma: SMAIndicator = SMAIndicator(close=time_series[self.column_names.close_price],
                                              window=self.sma_period,
                                              fillna=True)
        self.ema: EMAIndicator = EMAIndicator(close=time_series[self.column_names.close_price],
                                              window=self.ema_period,
                                              fillna=True)
        self.rsi: RSIIndicator = RSIIndicator(time_series[self.column_names.close_price],
                                              window=self.rsi_period,
                                              fillna=True)
        self.stoch: StochasticOscillator = StochasticOscillator(high=time_series[self.column_names.high_price],
                                                                low=time_series[self.column_names.low_price],
                                                                close=time_series[self.column_names.close_price],
                                                                window=self.stoch_window,
                                                                smooth_window=self.stoch_smooth_window,
                                                                fillna=True)
        self.vpt: VolumePriceTrendIndicator = VolumePriceTrendIndicator(close=time_series[self.column_names.close_price],
                                                                        volume=time_series[self.column_names.volume],
                                                                        fillna=True)

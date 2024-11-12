import pandas as pd
import yaml
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from backend.utils.feature_column_names import FeatureColumnNames


class TechnicalIndicatorProvider:
    def __init__(self, time_series: pd.DataFrame):
        """
        This class calculates several technical indicators like the EMA, MACD and more based on a pandas Dataframe time series of OHLC data.

        :param time_series: A time series of OHLC stock data
        """
        with open('../config.yaml', "r") as file:
            config = yaml.safe_load(file)
            technical_indicators_parameters = config["technical_indicator_parameters"]

        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.rounding_factor: int = config["calculation_parameters"].get('rounding_factor')
        self.time_series: pd.DataFrame = time_series

        self.adx_period: int = technical_indicators_parameters['adx_period']
        self.atr_window: int = technical_indicators_parameters['atr_window']
        self.bollinger_period: int = technical_indicators_parameters['bollinger_period']
        self.bollinger_std: int = technical_indicators_parameters['bollinger_std']
        self.ema_period: int = technical_indicators_parameters['ema_period']
        self.macd_short_period: int = technical_indicators_parameters['macd_short_period']
        self.macd_long_period: int = technical_indicators_parameters['macd_long_period_period']
        self.macd_signal_period: int = technical_indicators_parameters['macd_signal_period']
        self.rsi_period: int = technical_indicators_parameters['rsi_period']
        self.sma_short_period: int = technical_indicators_parameters['sma_short_period']
        self.sma_middle_period: int = technical_indicators_parameters['sma_middle_period']
        self.sma_long_period: int = technical_indicators_parameters['sma_long_period']

        self.adx: ADXIndicator = ADXIndicator(high=self.time_series[self.column_names.high_price],
                                              low=self.time_series[self.column_names.low_price],
                                              close=self.time_series[self.column_names.close_price],
                                              window=self.adx_period,
                                              fillna=True)
        self.atr: AverageTrueRange = AverageTrueRange(high=self.time_series[self.column_names.high_price],
                                                      low=self.time_series[self.column_names.low_price],
                                                      close=self.time_series[self.column_names.close_price],
                                                      window=self.atr_window,
                                                      fillna=True)

        self.bb: BollingerBands = BollingerBands(close=self.time_series[self.column_names.close_price],
                                                 window=self.bollinger_period,
                                                 window_dev=self.bollinger_std,
                                                 fillna=True)

        self.ema: EMAIndicator = EMAIndicator(close=self.time_series[self.column_names.close_price],
                                              window=self.ema_period,
                                              fillna=True)

        self.macd: MACD = MACD(close=self.time_series[self.column_names.close_price],
                               window_fast=self.macd_short_period,
                               window_slow=self.macd_long_period,
                               window_sign=self.macd_signal_period,
                               fillna=True)

        self.rsi: RSIIndicator = RSIIndicator(self.time_series[self.column_names.close_price],
                                              window=self.rsi_period,
                                              fillna=True)

        self.sma_short: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.close_price],
                                                    window=self.sma_short_period,
                                                    fillna=True)

        self.sma_middle: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.close_price],
                                                     window=self.sma_middle_period,
                                                     fillna=True)
        self.sma_long: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.close_price],
                                                   window=self.sma_long_period,
                                                   fillna=True)
        self.sma_slope: float = (self.sma_short.sma_indicator().diff() / self.sma_short_period)

        self.technical_indicators: pd.DataFrame = self.get_technical_indicators().round(self.rounding_factor)

    def get_technical_indicators(self) -> pd.DataFrame:
        """
        This function calculates several technical indicators and provides a pandas Dataframe with a time series
        of several selected technical indicators
        :return: A time series of several technical indicators based on OHLC stock data
        """
        technical_indicators: pd.DataFrame = pd.DataFrame()
        technical_indicators[self.column_names.adx]: pd.Series = self.adx.adx().round(self.rounding_factor)
        technical_indicators[self.column_names.atr]: pd.Series = self.atr.average_true_range().round(self.rounding_factor)
        technical_indicators[self.column_names.percent_b]: pd.Series = self.bb.bollinger_pband().round(self.rounding_factor)
        technical_indicators[self.column_names.ema_price]: pd.Series = self.ema.ema_indicator().round(self.rounding_factor)
        technical_indicators[self.column_names.ema_slope]: pd.Series = (self.ema.ema_indicator().diff()/self.ema_period).round(self.rounding_factor)
        technical_indicators[self.column_names.macd]: pd.Series = self.macd.macd().round(self.rounding_factor)
        technical_indicators[self.column_names.macd_signal]: pd.Series = self.macd.macd_signal().round(self.rounding_factor)
        technical_indicators[self.column_names.macd_hist]: pd.Series = self.macd.macd_diff().round(self.rounding_factor)
        technical_indicators[self.column_names.rsi]: pd.Series = self.rsi.rsi().round(self.rounding_factor)
        technical_indicators[self.column_names.sma_price]: pd.Series = self.sma_short.sma_indicator().round(self.rounding_factor)
        technical_indicators[self.column_names.sma_slope]: pd.Series = self.sma_slope
        return technical_indicators

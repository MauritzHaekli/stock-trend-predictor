import pandas as pd
import yaml
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from backend.src.models.feature_column_names import FeatureColumnNames


class TechnicalIndicatorProvider:
    def __init__(self, time_series: pd.DataFrame):
        """
        This class calculates several technical indicators like the EMA, MACD and more based on a pandas Dataframe time series of OHLC data.

        :param time_series: A time series of OHLC stock data
        """
        try:
            with open('../../config.yaml', "r") as file:
                config = yaml.safe_load(file)
                technical_indicators_parameters = config["technical_indicator_parameters"]
        except FileNotFoundError:
            raise FileNotFoundError("The config.yaml file was not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading YAML file: {e}")

        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.rounding_factor: int = config["calculation_parameters"].get('rounding_factor')
        self.time_series: pd.DataFrame = time_series

        self.adx_period: int = technical_indicators_parameters['adx_period']
        self.atr_window: int = technical_indicators_parameters['atr_window']
        self.bollinger_period: int = technical_indicators_parameters['bollinger_period']
        self.bollinger_std: int = technical_indicators_parameters['bollinger_std']
        self.ema_period: int = technical_indicators_parameters['ema_period']
        self.macd_short_period: int = technical_indicators_parameters['macd_short_period']
        self.macd_long_period: int = technical_indicators_parameters['macd_long_period']
        self.macd_signal_period: int = technical_indicators_parameters['macd_signal_period']
        self.rsi_period: int = technical_indicators_parameters['rsi_period']
        self.sma_short_period: int = technical_indicators_parameters['sma_short_period']
        self.sma_middle_period: int = technical_indicators_parameters['sma_middle_period']
        self.sma_long_period: int = technical_indicators_parameters['sma_long_period']

        self.adx: ADXIndicator = ADXIndicator(high=self.time_series[self.column_names.HIGH_PRICE],
                                              low=self.time_series[self.column_names.LOW_PRICE],
                                              close=self.time_series[self.column_names.CLOSE_PRICE],
                                              window=self.adx_period,
                                              fillna=True)
        self.atr: AverageTrueRange = AverageTrueRange(high=self.time_series[self.column_names.HIGH_PRICE],
                                                      low=self.time_series[self.column_names.LOW_PRICE],
                                                      close=self.time_series[self.column_names.CLOSE_PRICE],
                                                      window=self.atr_window,
                                                      fillna=True)

        self.bb: BollingerBands = BollingerBands(close=self.time_series[self.column_names.CLOSE_PRICE],
                                                 window=self.bollinger_period,
                                                 window_dev=self.bollinger_std,
                                                 fillna=True)

        self.ema: EMAIndicator = EMAIndicator(close=self.time_series[self.column_names.CLOSE_PRICE],
                                              window=self.ema_period,
                                              fillna=True)

        self.macd: MACD = MACD(close=self.time_series[self.column_names.CLOSE_PRICE],
                               window_fast=self.macd_short_period,
                               window_slow=self.macd_long_period,
                               window_sign=self.macd_signal_period,
                               fillna=True)

        self.rsi: RSIIndicator = RSIIndicator(self.time_series[self.column_names.CLOSE_PRICE],
                                              window=self.rsi_period,
                                              fillna=True)

        self.sma_short: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.CLOSE_PRICE],
                                                    window=self.sma_short_period,
                                                    fillna=True)

        self.sma_middle: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.CLOSE_PRICE],
                                                     window=self.sma_middle_period,
                                                     fillna=True)
        self.sma_long: SMAIndicator = SMAIndicator(close=self.time_series[self.column_names.CLOSE_PRICE],
                                                   window=self.sma_long_period,
                                                   fillna=True)
        self.sma_slope: float = self.sma_short.sma_indicator().diff() / self.sma_short_period

        self.technical_indicators: pd.DataFrame = self.get_technical_indicators()

    def get_technical_indicators(self) -> pd.DataFrame:
        """
        This function calculates several technical indicators and provides a pandas Dataframe with a time series
        of several selected technical indicators
        :return: A time series of several technical indicators based on OHLC stock data
        """
        technical_indicators: pd.DataFrame = pd.DataFrame()
        technical_indicators[self.column_names.ADX]: pd.Series = self.adx.adx()
        technical_indicators[self.column_names.ATR]: pd.Series = self.atr.average_true_range()
        technical_indicators[self.column_names.PERCENT_B]: pd.Series = self.bb.bollinger_pband()
        technical_indicators[self.column_names.EMA]: pd.Series = self.ema.ema_indicator()
        technical_indicators[self.column_names.EMA_SLOPE]: pd.Series = (self.ema.ema_indicator().diff()/self.ema_period)
        technical_indicators[self.column_names.MACD]: pd.Series = self.macd.macd()
        technical_indicators[self.column_names.MACD_SIGNAL]: pd.Series = self.macd.macd_signal()
        technical_indicators[self.column_names.MACD_HIST]: pd.Series = self.macd.macd_diff()
        technical_indicators[self.column_names.RSI]: pd.Series = self.rsi.rsi()
        technical_indicators[self.column_names.SMA]: pd.Series = self.sma_short.sma_indicator()
        technical_indicators[self.column_names.SMA_SLOPE]: pd.Series = self.sma_slope
        rounded_technical_indicators: pd.DataFrame = technical_indicators.round(decimals=self.rounding_factor)
        return rounded_technical_indicators

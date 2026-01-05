import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from backend.src.schema.ohlcv import OHLCVColumns
from backend.src.schema.technical_indicators import TechnicalIndicatorColumns

class TechnicalIndicatorProvider:
    def __init__(self, time_series: pd.DataFrame, params: dict):
        """
        This class calculates several technical indicators like the EMA, MACD and more based on a pandas Dataframe time series of OHLC data.

        :param time_series: A time series of OHLC stock data
        :param params: A dictionary of params to pass to the individual technical indicator
        """

        self.params: dict = params
        self.technical_indicators_parameters = self.params.get("technical_indicator_parameters")

        self.rounding_factor: int = self.params.get("calculation_parameters").get('rounding_factor')
        self.raw_ohlcv_columns: OHLCVColumns = OHLCVColumns()
        self.technical_indicator_columns: TechnicalIndicatorColumns = TechnicalIndicatorColumns()
        self.time_series: pd.DataFrame = time_series

        self._validate_input()

        self.open_price: pd.Series = time_series[self.raw_ohlcv_columns.OPEN]
        self.high_price: pd.Series = time_series[self.raw_ohlcv_columns.HIGH]
        self.low_price: pd.Series = time_series[self.raw_ohlcv_columns.LOW]
        self.close_price: pd.Series = time_series[self.raw_ohlcv_columns.CLOSE]
        self.volume: pd.Series = time_series[self.raw_ohlcv_columns.VOLUME]

        self.adx_period: int = self.technical_indicators_parameters.get('adx_period')
        self.atr_window: int = self.technical_indicators_parameters.get('atr_window')
        self.bollinger_period: int = self.technical_indicators_parameters.get('bollinger_period')
        self.bollinger_std: int = self.technical_indicators_parameters.get('bollinger_std')
        self.ema_period: int = self.technical_indicators_parameters.get('ema_period')
        self.macd_short_period: int = self.technical_indicators_parameters.get('macd_short_period')
        self.macd_long_period: int = self.technical_indicators_parameters.get('macd_long_period')
        self.macd_signal_period: int = self.technical_indicators_parameters.get('macd_signal_period')
        self.rsi_period: int = self.technical_indicators_parameters.get('rsi_period')
        self.sma_period: int = self.technical_indicators_parameters.get('sma_period')

        self.adx_provider: ADXIndicator = ADXIndicator(high=self.high_price,
                                                       low=self.low_price,
                                                       close=self.close_price,
                                                       window=self.adx_period,
                                                       fillna=False)
        self.atr_provider: AverageTrueRange = AverageTrueRange(high=self.high_price,
                                                               low=self.low_price,
                                                               close=self.close_price,
                                                               window=self.atr_window,
                                                               fillna=False)

        self.bb_provider: BollingerBands = BollingerBands(close=self.close_price,
                                                          window=self.bollinger_period,
                                                          window_dev=self.bollinger_std,
                                                          fillna=False)

        self.ema_provider: EMAIndicator = EMAIndicator(close=self.close_price,
                                                       window=self.ema_period,
                                                       fillna=False)

        self.macd_provider: MACD = MACD(close=self.close_price,
                                        window_fast=self.macd_short_period,
                                        window_slow=self.macd_long_period,
                                        window_sign=self.macd_signal_period,
                                        fillna=False)

        self.rsi_provider: RSIIndicator = RSIIndicator(self.close_price,
                                                       window=self.rsi_period,
                                                       fillna=False)

        self.sma_provider: SMAIndicator = SMAIndicator(close=self.close_price,
                                                       window=self.sma_period,
                                                       fillna=False)

        self._technical_indicators: pd.DataFrame | None = None

    def get_indicator(self, col: str) -> pd.Series:
        return self.technical_indicators[col]

    @property
    def ema(self) -> pd.Series:
        return self.get_indicator(self.technical_indicator_columns.EMA)

    @property
    def atr(self) -> pd.Series:
        return self.get_indicator(self.technical_indicator_columns.ATR)

    @property
    def bollinger_percent(self) -> pd.Series:
        return self.get_indicator(self.technical_indicator_columns.BOLLINGER_B)

    @property
    def technical_indicators(self) -> pd.DataFrame:
        if self._technical_indicators is None:
            self._technical_indicators = self._compute_technical_indicators()
        return self._technical_indicators

    def _compute_technical_indicators(self) -> pd.DataFrame:
        """
        This function calculates several technical indicators and provides a pandas Dataframe with a time series
        of several selected technical indicators
        :return: A time series of several technical indicators based on OHLC stock data
        """
        technical_indicators: pd.DataFrame = pd.DataFrame(index=self.time_series.index)
        technical_indicators[self.technical_indicator_columns.ADX]: pd.Series = self.adx_provider.adx()
        technical_indicators[self.technical_indicator_columns.ATR]: pd.Series = self.atr_provider.average_true_range()
        technical_indicators[self.technical_indicator_columns.BOLLINGER_B]: pd.Series = self.bb_provider.bollinger_pband()
        technical_indicators[self.technical_indicator_columns.EMA]: pd.Series = self.ema_provider.ema_indicator()
        technical_indicators[self.technical_indicator_columns.MACD]: pd.Series = self.macd_provider.macd()
        technical_indicators[self.technical_indicator_columns.RSI]: pd.Series = self.rsi_provider.rsi()
        technical_indicators[self.technical_indicator_columns.SMA]: pd.Series = self.sma_provider.sma_indicator()
        rounded_technical_indicators: pd.DataFrame = technical_indicators.round(decimals=self.rounding_factor)
        return rounded_technical_indicators

    def _validate_input(self):
        required = {
            self.raw_ohlcv_columns.OPEN,
            self.raw_ohlcv_columns.HIGH,
            self.raw_ohlcv_columns.LOW,
            self.raw_ohlcv_columns.CLOSE,
            self.raw_ohlcv_columns.VOLUME,
        }
        missing = required - set(self.time_series.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

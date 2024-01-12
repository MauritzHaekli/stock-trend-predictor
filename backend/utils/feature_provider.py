import pandas as pd
from feature_column_names import FeatureColumnNames
from technical_indicator_provider import TechnicalIndicatorProvider


class FeatureCalculator:
    def __init__(self, time_series: pd.DataFrame):

        """
        This class is used to add features to a time series of open-, high-, low- and close-stock data.
        :param time_series:
        """

        self.trend_shift: int = 1
        self.decimal_place: int = 4
        self.cutoff: int = 30

        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.technical_indicators: TechnicalIndicatorProvider = TechnicalIndicatorProvider(time_series)

        self.time_series: pd.DataFrame = time_series
        self.feature_time_series: pd.DataFrame = self.get_feature_time_series(self.time_series)

    def get_feature_time_series(self, feature_data: pd.DataFrame) -> pd.DataFrame:

        feature_data: pd.DataFrame = feature_data.copy()

        feature_data[self.column_names.percent_b]: pd.Series = self.get_bollinger_percent()
        feature_data[self.column_names.atr]: pd.Series = self.technical_indicators.atr.average_true_range().round(self.decimal_place)
        feature_data[self.column_names.macd]: pd.Series = self.technical_indicators.macd.macd().round(self.decimal_place)
        feature_data[self.column_names.macd_signal]: pd.Series = self.technical_indicators.macd.macd_signal().round(self.decimal_place)
        feature_data[self.column_names.macd_hist]: pd.Series = self.technical_indicators.macd.macd_diff().round(self.decimal_place)
        feature_data[self.column_names.adx]: pd.Series = self.technical_indicators.adx.adx().round(self.decimal_place)
        feature_data[self.column_names.sma]: pd.Series = self.technical_indicators.sma.sma_indicator().round(self.decimal_place)
        feature_data[self.column_names.ema]: pd.Series = self.technical_indicators.ema.ema_indicator().round(self.decimal_place)
        feature_data[self.column_names.rsi]: pd.Series = self.technical_indicators.rsi.rsi().round(self.decimal_place)
        feature_data[self.column_names.fast_stochastic]: pd.Series = self.technical_indicators.stoch.stoch().round(self.decimal_place)
        feature_data[self.column_names.slow_stochastic]: pd.Series = self.technical_indicators.stoch.stoch_signal().round(self.decimal_place)
        feature_data[self.column_names.vpt]: pd.Series = self.technical_indicators.vpt.volume_price_trend().round(self.decimal_place)

        feature_data.index = pd.to_datetime(feature_data['datetime'], format='%Y-%m-%d %H:%M:%S')
        feature_data['day']: pd.Series = feature_data.index.day_of_week
        feature_data['hour']: pd.Series = feature_data.index.hour

        feature_data['open change']: pd.Series = self.get_price_change(self.column_names.open_price)
        feature_data['high change']: pd.Series = self.get_price_change(self.column_names.high_price)
        feature_data['low change']: pd.Series = self.get_price_change(self.column_names.low_price)
        feature_data['close change']: pd.Series = self.get_price_change(self.column_names.close_price)

        feature_data['price movement']: pd.Series = self.get_price_difference(self.column_names.open_price, self.column_names.close_price)
        feature_data['price range']: pd.Series = self.get_price_difference(self.column_names.low_price, self.column_names.high_price)
        feature_data['price trend']: pd.Series = self.get_current_trend()

        feature_data['open trend']: pd.Series = self.get_recent_trend(self.column_names.open_price, self.trend_shift)
        feature_data['high trend']: pd.Series = self.get_recent_trend(self.column_names.open_price, self.trend_shift)
        feature_data['low trend']: pd.Series = self.get_recent_trend(self.column_names.open_price, self.trend_shift)
        feature_data['close trend']: pd.Series = self.get_recent_trend(self.column_names.open_price, self.trend_shift)
        feature_data['volume trend']: pd.Series = self.get_recent_trend(self.column_names.open_price, self.trend_shift)

        feature_data_cutoff = feature_data[self.cutoff:]

        return feature_data_cutoff

    def get_bollinger_percent(self) -> pd.Series:
        return ((self.time_series[self.column_names.open_price] - self.technical_indicators.bb.bollinger_lband()) / (self.technical_indicators.bb.bollinger_hband() - self.technical_indicators.bb.bollinger_lband())).round(self.decimal_place)

    def get_price_change(self, price_name: str) -> pd.Series:
        return self.time_series[price_name].pct_change().round(self.decimal_place)

    def get_price_difference(self, starting_price: str, final_price: str) -> pd.Series:
        return (self.time_series[final_price] - self.time_series[starting_price]).round(self.decimal_place)

    def get_current_trend(self) -> pd.Series:
        current_price_differences: pd.Series = self.get_price_difference(self.column_names.open_price, self.column_names.close_price)
        return (current_price_differences > 0).astype(int)

    def get_recent_trend(self, trend_name: str, trend_length) -> pd.Series:
        return (self.time_series[trend_name] > self.time_series[trend_name].shift(trend_length)).astype(int)

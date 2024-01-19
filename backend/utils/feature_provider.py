import pandas as pd
from backend.utils.feature_column_names import FeatureColumnNames
from backend.utils.trend_provider import TrendProvider
from backend.utils.technical_indicator_provider import TechnicalIndicatorProvider
from backend.utils.datetime_provider import DatetimeProvider


class FeatureProvider:
    def __init__(self, time_series: pd.DataFrame, periods: int = 1, rounding_factor: int = 4, cutoff: int = 30):

        """
        This class is used to add features to a time series of OHLC stock data.
        The features range from a variety of technical indicators and datetime information to latest change- and trend-data for OHLC prices and volume.
        :param time_series: A time series of open-, high-, low- and close-stock data.
        :param periods: An integer used as a pointer to fetch information for the current iteration from the former iteration a trend shift away.
        :param rounding_factor: Used to round several features.
        :param cutoff: Since the first entries in time_series are filled with NaN values during calculation of technical indicators, we need to cut off the first entries with a length of self.cutoff
        :param self.column_names: The names of feature columns provided by the FeatureColumnNames class.
        :param self.feature_time_series: A dataframe containing OHLC stock data and all features calculated in self.get_feature_time_series()
        """
        self.column_names: FeatureColumnNames = FeatureColumnNames()

        if not isinstance(time_series, pd.DataFrame):
            raise ValueError("The 'time_series' parameter must be a pandas DataFrame.")

        required_columns = [self.column_names.datetime, self.column_names.open_price, self.column_names.high_price, self.column_names.low_price, self.column_names.close_price, self.column_names.volume]
        missing_columns = [time_series_column for time_series_column in required_columns if time_series_column not in time_series.columns]
        if missing_columns:
            raise ValueError(
                f"The 'time_series' DataFrame is missing the following required columns: {missing_columns}")

        self.time_series = time_series
        self.periods = periods
        self.rounding_factor: int = rounding_factor
        self.cutoff: int = cutoff

        self.feature_time_series: pd.DataFrame = self.get_feature_time_series(time_series, rounding_factor)

    def get_feature_time_series(self, time_series: pd.DataFrame, rounding_factor: int) -> pd.DataFrame:

        """
        Adds all the additional features like technical indicators, daytime information, price changes and price trends to the OHLC time series.
        :param time_series: A time series of OHLC data.
        :param rounding_factor: A factor used for rounding all calculations.
        :return: A pandas Dataframe containing a time series of feature data in addition to OHLC price data.
        """

        feature_data: pd.DataFrame = pd.DataFrame()
        feature_data[self.column_names.datetime] = time_series[self.column_names.datetime]
        price_columns: [str] = [self.column_names.open_price, self.column_names.high_price, self.column_names.low_price, self.column_names.close_price]
        price_trend: TrendProvider = TrendProvider(self.time_series, self.rounding_factor)

        for price_column in price_columns:
            feature_data[price_column] = time_series[price_column]
            feature_data[f"{price_column} change"]: pd.Series = price_trend.get_absolute_change(price_column, self.periods)
            feature_data[f"{price_column} change (%)"]: pd.Series = price_trend.get_percentage_change(price_column, self.periods)
            feature_data[f"{price_column} trend"]: pd.Series = price_trend.get_recent_trend(price_column, self.periods)

        feature_data[self.column_names.volume] = time_series[self.column_names.volume]

        average_trend: TrendProvider = TrendProvider(feature_data, self.rounding_factor)
        technical_indicators: TechnicalIndicatorProvider = TechnicalIndicatorProvider(feature_data, rounding_factor)
        average_columns: [str] = [self.column_names.sma_price, self.column_names.ema_price]

        for average_column in average_columns:
            feature_data[average_column]: pd.Series = technical_indicators.sma.sma_indicator().round(rounding_factor)
            feature_data[f"{average_column} change"]: pd.Series = average_trend.get_absolute_change(average_column, self.periods)
            feature_data[f"{average_column} change(%)"]: pd.Series = average_trend.get_percentage_change(average_column, self.periods)
            feature_data[f"{average_column} trend"]: pd.Series = average_trend.get_recent_trend(average_column, self.periods)

        feature_data[self.column_names.price_movement]: pd.Series = price_trend.get_column_difference(self.column_names.open_price, self.column_names.close_price)
        feature_data[self.column_names.price_range]: pd.Series = price_trend.get_column_difference(self.column_names.low_price, self.column_names.high_price)
        feature_data[self.column_names.price_trend]: pd.Series = price_trend.get_current_trend(self.column_names.open_price, self.column_names.close_price)

        feature_data[self.column_names.percent_b]: pd.Series = technical_indicators.bb.bollinger_pband()
        feature_data[self.column_names.atr]: pd.Series = technical_indicators.atr.average_true_range().round(rounding_factor)
        feature_data[self.column_names.macd]: pd.Series = technical_indicators.macd.macd().round(rounding_factor)
        feature_data[self.column_names.macd_signal]: pd.Series = technical_indicators.macd.macd_signal().round(rounding_factor)
        feature_data[self.column_names.macd_hist]: pd.Series = technical_indicators.macd.macd_diff().round(rounding_factor)
        feature_data[self.column_names.adx]: pd.Series = technical_indicators.adx.adx().round(rounding_factor)
        feature_data[self.column_names.rsi]: pd.Series = technical_indicators.rsi.rsi().round(rounding_factor)
        feature_data[self.column_names.fast_stochastic]: pd.Series = technical_indicators.stoch.stoch().round(rounding_factor)
        feature_data[self.column_names.slow_stochastic]: pd.Series = technical_indicators.stoch.stoch_signal().round(rounding_factor)
        feature_data[self.column_names.vpt]: pd.Series = technical_indicators.vpt.volume_price_trend().round(rounding_factor)

        datetime: DatetimeProvider = DatetimeProvider(feature_data)

        feature_data[self.column_names.day]: pd.Series = datetime.day_series
        feature_data[self.column_names.hour]: pd.Series = datetime.hour_series

        feature_data_cutoff: pd.DataFrame = feature_data[self.cutoff:]

        return feature_data_cutoff


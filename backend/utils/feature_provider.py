import pandas as pd
from backend.utils.feature_column_names import FeatureColumnNames
from backend.utils.feature_change_calculator import FeatureChangeCalculator
from backend.utils.technical_indicator_provider import TechnicalIndicatorProvider
from backend.utils.datetime_provider import DatetimeProvider


class FeatureProvider:
    def __init__(self, time_series: pd.DataFrame, periods: int = 9, rounding_factor: int = 4, cutoff: int = 30):

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
        self.technical_indicator_provider: TechnicalIndicatorProvider = TechnicalIndicatorProvider(time_series)

        if not isinstance(time_series, pd.DataFrame):
            raise ValueError("The 'time_series' parameter must be a pandas DataFrame.")

        required_columns = [self.column_names.datetime, self.column_names.open_price, self.column_names.high_price, self.column_names.low_price, self.column_names.close_price, self.column_names.volume]
        missing_columns = [time_series_column for time_series_column in required_columns if time_series_column not in time_series.columns]
        if missing_columns:
            raise ValueError(
                f"The 'time_series' DataFrame is missing the following required columns: {missing_columns}")

        self.time_series = time_series
        self.technical_indicators: pd.DataFrame = self.technical_indicator_provider.technical_indicators
        self.rounding_factor: int = rounding_factor
        self.periods = periods
        self.cutoff: int = cutoff

        self.feature_time_series: pd.DataFrame = self.get_feature_time_series(self.time_series, self.technical_indicators)

    def get_feature_time_series(self, time_series: pd.DataFrame, technical_indicators: pd.DataFrame) -> pd.DataFrame:

        """
        Adds all the additional features like technical indicators, daytime information, price changes and price trends to the OHLC time series.
        :param technical_indicators:
        :param time_series: .
        :return: A pandas Dataframe containing a time series of feature data, consisting of OHLC price data, technical indicators, price-, trend- and datetime information.
        """

        open_price: pd.Series = time_series[self.column_names.open_price]
        high_price: pd.Series = time_series[self.column_names.high_price]
        low_price: pd.Series = time_series[self.column_names.low_price]
        close_price: pd.Series = time_series[self.column_names.close_price]
        volume: pd.Series = time_series[self.column_names.volume]
        sma_price: pd.Series = technical_indicators[self.column_names.sma_price]
        sma_slope: pd.Series = technical_indicators[self.column_names.sma_slope]
        ema_price: pd.Series = technical_indicators[self.column_names.ema_price]
        ema_slope: pd.Series = technical_indicators[self.column_names.ema_slope]
        percent_b: pd.Series = technical_indicators[self.column_names.percent_b]
        rsi: pd.Series = technical_indicators[self.column_names.rsi]
        adx: pd.Series = technical_indicators[self.column_names.adx]

        datetime_provider: DatetimeProvider = DatetimeProvider(time_series)
        open_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(open_price, self.periods, self.rounding_factor)
        high_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(high_price, self.periods, self.rounding_factor)
        low_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(low_price, self.periods, self.rounding_factor)
        close_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(close_price, self.periods, self.rounding_factor)
        volume_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(volume, self.periods, self.rounding_factor)
        sma_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(sma_price, self.periods, self.rounding_factor)
        sma_slope_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(sma_slope, self.periods, self.rounding_factor)
        ema_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(ema_price, self.periods, self.rounding_factor)
        ema_slope_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(ema_slope, self.periods, self.rounding_factor)
        percent_b_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(percent_b, self.periods, self.rounding_factor)
        rsi_feature_provider: FeatureChangeCalculator = FeatureChangeCalculator(rsi, self.periods, self.rounding_factor)

        open_latest_absolute_change: pd.Series = open_feature_provider.latest_absolute_change
        open_recent_absolute_change: pd.Series = open_feature_provider.recent_absolute_change

        open_latest_percentage_change: pd.Series = open_feature_provider.latest_percentage_change
        open_recent_percentage_change: pd.Series = open_feature_provider.recent_percentage_change

        open_latest_trend_change: pd.Series = open_feature_provider.latest_trend_change
        open_recent_trend_change: pd.Series = open_feature_provider.recent_trend_change

        open_information: pd.DataFrame = pd.concat([open_price,
                                                    open_latest_absolute_change,
                                                    open_recent_absolute_change,
                                                    open_latest_percentage_change,
                                                    open_recent_percentage_change,
                                                    open_latest_trend_change,
                                                    open_recent_trend_change,
                                                    ], axis=1)

        feature_data: pd.DataFrame = pd.DataFrame()
        feature_data[self.column_names.datetime] = self.time_series[self.column_names.datetime]

        concat_dataframes: [pd.DataFrame] = [feature_data,
                                             open_information,
                                             high_price,
                                             low_price,
                                             close_price,
                                             sma_price,
                                             sma_slope,
                                             ema_price,
                                             ema_slope,
                                             percent_b,
                                             rsi,
                                             adx
                                             ]
        feature_data = pd.concat(concat_dataframes, axis=1)

        feature_data[self.column_names.day]: pd.Series = datetime_provider.day_series
        feature_data[self.column_names.hour]: pd.Series = datetime_provider.hour_series

        feature_data_cutoff: pd.DataFrame = feature_data[self.cutoff:]

        return feature_data_cutoff






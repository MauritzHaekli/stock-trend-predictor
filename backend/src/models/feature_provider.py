import pandas as pd
from backend.src.models.feature_column_names import FeatureColumnNames
from backend.src.models.feature_change_calculator import FeatureChangeCalculator
from backend.src.models.technical_indicator_provider import TechnicalIndicatorProvider
from backend.src.models.datetime_provider import DatetimeProvider


class FeatureProvider:
    """
    Enhances a time series DataFrame containing OHLC (Open, High, Low, Close) stock data by adding calculated features.

    Features include:
        - Technical indicators (e.g., SMA, EMA, RSI).
        - Trend and change features for OHLC prices and volume.
        - Datetime-based features like day and hour of each record.

    Parameters:
        time_series (pd.DataFrame): DataFrame containing stock OHLC and volume data.
        periods (int): Number of periods for change and trend calculations (default: 9).
        rounding_factor (int): Decimal places to round feature values (default: 4).
        cutoff (int): Number of initial rows to exclude due to NaN values from calculations (default: 30).
    """

    def __init__(self, time_series: pd.DataFrame, periods: int = 9, rounding_factor: int = 4, cutoff: int = 40):

        self.column_names: FeatureColumnNames = FeatureColumnNames()
        self.technical_indicator_provider: TechnicalIndicatorProvider = TechnicalIndicatorProvider(time_series)
        self.time_series: pd.DataFrame = time_series
        self.periods: int = periods
        self.rounding_factor: int = rounding_factor
        self.cutoff: int = cutoff

        self.technical_indicators: pd.DataFrame = self.technical_indicator_provider.technical_indicators
        self.feature_time_series: pd.DataFrame = self.build_feature_time_series()

    def build_feature_time_series(self) -> pd.DataFrame:
        """
        Constructs the feature-enriched time series with additional technical indicators, datetime features, and change metrics.

        :return: DataFrame with original OHLC data, technical indicators, and calculated features.
        """
        feature_data = pd.DataFrame()
        feature_data[self.column_names.DATETIME] = self.time_series[self.column_names.DATETIME]

        ohlc_features = {
            self.column_names.OPEN_PRICE: self.time_series[self.column_names.OPEN_PRICE],
            self.column_names.HIGH_PRICE: self.time_series[self.column_names.HIGH_PRICE],
            self.column_names.LOW_PRICE: self.time_series[self.column_names.LOW_PRICE],
            self.column_names.CLOSE_PRICE: self.time_series[self.column_names.CLOSE_PRICE],
            self.column_names.VOLUME: self.time_series[self.column_names.VOLUME],
        }

        for feature_name, feature_series in ohlc_features.items():
            feature_data[feature_name] = feature_series

        tech_indicators: list[str] = [self.column_names.ADX,
                                      self.column_names.ATR,
                                      self.column_names.SMA,
                                      self.column_names.EMA,
                                      self.column_names.RSI,
                                      self.column_names.MACD_SIGNAL,
                                      self.column_names.PERCENT_B]

        for indicator in tech_indicators:
            feature_data[indicator] = self.technical_indicators[indicator].shift(1)

        selected_columns = []

        for feature_series in selected_columns:
            feature_data = self.add_feature_change_columns(feature_data, feature_data[feature_series], self.periods)

        datetime_provider = DatetimeProvider(self.time_series)
        feature_data[self.column_names.DAY] = datetime_provider.get_day_series()
        feature_data[self.column_names.HOUR] = datetime_provider.get_hour_series()

        return feature_data[self.cutoff:]

    def add_feature_change_columns(self, feature_data: pd.DataFrame,
                                   series: pd.Series, periods: int) -> pd.DataFrame:
        """
        Adds latest and recent change columns for absolute, percentage, and trend changes of a given feature.

        :param feature_data: DataFrame to which the new columns will be added.
        :param series: Series of the feature values.
        :param periods: The number of periods we go back to calculate a change for
        :return: DataFrame with added feature change columns.
        """
        feature_calculator = FeatureChangeCalculator(series, self.rounding_factor)
        series_name = series.name
        for period in range(1, periods + 1):
            feature_absolute_change_name: str = f"{series_name}_({period}_change)"
            feature_relative_change_name: str = f"{series_name}_({period}_change(%))"
            feature_trend_change_name: str = f"{series_name}_({period}_trend)"
            feature_data[feature_absolute_change_name] = feature_calculator.get_absolute_change(period)
            feature_data[feature_relative_change_name] = feature_calculator.get_relative_change(period)
            feature_data[feature_trend_change_name] = feature_calculator.get_trend_change(period)

        return feature_data

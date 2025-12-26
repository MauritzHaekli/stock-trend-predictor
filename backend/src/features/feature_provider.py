import pandas as pd
from backend.src.features.technical_indicator_provider import TechnicalIndicatorProvider
from backend.src.features.datetime_provider import DatetimeProvider
from backend.src.schema.raw_ohlcv import RawOHLCVColumns


class FeatureProvider:
    """
    Builds the full feature matrix used for modeling by combining
    multiple feature sources (OHLCV, technical indicators, etc.).
    """

    def __init__(self, time_series: pd.DataFrame, params: dict, cutoff: int = 40):
        self.time_series = time_series
        self.params = params
        self.cutoff = cutoff

        self.raw_ohlcv_columns = RawOHLCVColumns()

        self.technical_indicator_provider = TechnicalIndicatorProvider(time_series=time_series,params=params)
        self.datetime_provider = DatetimeProvider(self.time_series)

        self.feature_time_series = self._build_feature_time_series()

    def _build_feature_time_series(self) -> pd.DataFrame:
        """
        Combine all feature blocks into a single DataFrame.
        """

        ohlcv_features = pd.DataFrame(
            {
                self.raw_ohlcv_columns.OPEN: self.time_series[self.raw_ohlcv_columns.OPEN],
                self.raw_ohlcv_columns.HIGH: self.time_series[self.raw_ohlcv_columns.HIGH],
                self.raw_ohlcv_columns.LOW: self.time_series[self.raw_ohlcv_columns.LOW],
                self.raw_ohlcv_columns.CLOSE: self.time_series[self.raw_ohlcv_columns.CLOSE],
                self.raw_ohlcv_columns.VOLUME: self.time_series[self.raw_ohlcv_columns.VOLUME],
            },
            index=self.time_series.index,
        )

        technical_indicators: pd.DataFrame= self.technical_indicator_provider.technical_indicators

        datetime_features = self.datetime_provider.day_and_hour()


        feature_data = pd.concat(
            [
                ohlcv_features,
                technical_indicators,
                datetime_features
            ],
            axis=1,
        )

        feature_data = feature_data.iloc[self.cutoff:]

        return feature_data


import pandas as pd
from backend.src.schema.ohlcv import OHLCVColumns

class OHLCVProvider:
    """
    Provides OHLCV features from a price time series.
    """

    def __init__(self, time_series: pd.DataFrame):
        self.time_series = time_series
        self.ohlcv_columns = OHLCVColumns()

        self._validate_time_series()

    @property
    def ohlcv_features(self) -> pd.DataFrame:
        return self.time_series[self._required_columns].copy()

    @property
    def _required_columns(self) -> list[str]:
        return [
            self.ohlcv_columns.OPEN,
            self.ohlcv_columns.HIGH,
            self.ohlcv_columns.LOW,
            self.ohlcv_columns.CLOSE,
            self.ohlcv_columns.VOLUME,
        ]

    def _validate_time_series(self) -> None:
        """
        Validates that the input DataFrame contains all required OHLCV columns.
        """
        missing = set(self._required_columns) - set(self.time_series.columns)
        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {sorted(missing)}"
            )


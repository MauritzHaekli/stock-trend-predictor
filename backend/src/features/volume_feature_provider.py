import pandas as pd
import numpy as np
from backend.src.schema.volume_features import VolumeFeatureColumns

class VolumeFeatureProvider:

    def __init__(self, close_price: pd.Series, volume: pd.Series, atr: pd.Series, rounding_factor: int | None = 4):
        self.close_price = close_price
        self.volume = volume
        self.atr = atr
        self.rounding_factor = rounding_factor

        self.volume_feature_columns = VolumeFeatureColumns()

        self._validate_input()

    @property
    def volume_features(self) -> pd.DataFrame:
        """
        Return all return features as a single DataFrame.
        """
        volume_features = {
            self.volume_feature_columns.LOG_VOLUME: self.log_volume,
            self.volume_feature_columns.OBV: self.on_balance_volume,
            self.volume_feature_columns.OBV_SLOPE: self.get_obv_slope(),
            self.volume_feature_columns.OBV_SLOPE_ATR: self.get_obv_slope_atr(),
            self.volume_feature_columns.DOLLAR_VOLUME: self.dollar_volume,
        }

        return pd.DataFrame(volume_features, index=self.close_price.index)

    @property
    def log_volume(self) -> pd.Series:
        """
        Log Volume feature.
        """
        log_volume = np.log(self.volume.replace(0, np.nan))
        log_volume.name = self.volume_feature_columns.LOG_VOLUME

        if self.rounding_factor is not None:
            log_volume = log_volume.round(self.rounding_factor)
        return log_volume


    @property
    def on_balance_volume(self) -> pd.Series:
        """
        On-Balance Volume (OBV)
        """
        direction = np.sign(self.close_price.diff()).fillna(0)
        obv = (direction * self.volume).cumsum()
        obv.name = self.volume_feature_columns.OBV

        if self.rounding_factor is not None:
            obv = obv.round(self.rounding_factor)
        return obv

    def get_obv_slope(self, window: int = 20) -> pd.Series:
        """
        OBV slope
        """
        obv_slope_window = self.on_balance_volume.diff(window)
        obv_slope_window.name = self.volume_feature_columns.OBV_SLOPE

        if self.rounding_factor is not None:
            obv_slope_window = obv_slope_window.round(self.rounding_factor)
        return obv_slope_window

    def get_obv_slope_atr(self, window: int = 20) -> pd.Series:
        """
        OBV slope normalized by ATR.
        """
        obv_slope_atr_window = self.on_balance_volume.diff(window) / self.atr.replace(0, np.nan)
        obv_slope_atr_window.name = self.volume_feature_columns.OBV_SLOPE_ATR

        if self.rounding_factor is not None:
            obv_slope_atr_window = obv_slope_atr_window.round(self.rounding_factor)
        return obv_slope_atr_window

    @property
    def dollar_volume(self) -> pd.Series:
        """
        Dollar Volume
        """

        dollar_volume = self.close_price * self.volume
        dollar_volume.name = self.volume_feature_columns.DOLLAR_VOLUME

        if self.rounding_factor is not None:
            dollar_volume = dollar_volume.round(self.rounding_factor)

        return dollar_volume

    def _validate_input(self) -> None:
        if not self.close_price.index.equals(self.volume.index):
            raise ValueError("close_price and volume indices must match")

        if not pd.api.types.is_numeric_dtype(self.volume):
            raise TypeError("volume must be numeric")

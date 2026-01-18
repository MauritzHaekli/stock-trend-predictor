import pandas as pd
from backend.src.schema.price_features import PriceFeatureColumns
import numpy as np


class PriceFeatureProvider:

    def __init__(self, open_price: pd.Series, high_price: pd.Series, low_price: pd.Series, close_price: pd.Series, atr: pd.Series):
        self.open_price: pd.Series = open_price
        self.high_price: pd.Series = high_price
        self.low_price: pd.Series = low_price
        self.close_price: pd.Series = close_price
        self.atr: pd.Series = atr
        self.rounding_factor: int | None = None

        self.price_feature_columns = PriceFeatureColumns()

        self._validate_input()

    @property
    def price_features(self) -> pd.DataFrame:
        """
        Return all price features as a single DataFrame.
        """
        price_features = {
            self.price_feature_columns.PRICE_RANGE: self.price_range,
            self.price_feature_columns.PRICE_RANGE_ATR: self.price_range_atr,
            self.price_feature_columns.PRICE_BODY_ATR: self.price_body_atr,
            self.price_feature_columns.UPPER_WICK_ATR: self.price_upper_wick_atr,
            self.price_feature_columns.LOWER_WICK_ATR: self.price_lower_wick_atr,
            self.price_feature_columns.PRICE_CLOSE_LOCATION: self.price_close_location,
            self.price_feature_columns.PRICE_DIST_HH_ATR: self.price_dist_hh_atr(),
            self.price_feature_columns.PRICE_DIST_LL_ATR: self.price_dist_ll_atr(),

        }

        return pd.DataFrame(price_features, index=self.close_price.index)

    @property
    def price_range(self) -> pd.Series:
        """
        Calculates the range of a price (H_t - L_t)
        """

        price_range = self._price_range
        price_range.name = self.price_feature_columns.PRICE_RANGE

        return self._apply_rounding(price_range)

    @property
    def price_range_atr(self) -> pd.Series:
        """
        Calculates the candle range of a price, normalized by the atr ((H_t - L_t)/ATR_t)
        """

        price_range_atr = self._price_range / self._safe_atr
        price_range_atr.name = self.price_feature_columns.PRICE_RANGE_ATR

        return self._apply_rounding(price_range_atr)

    @property
    def price_body_atr(self) -> pd.Series:
        """
        Calculates the candle body size normalized by ATR (|C_t - O_t| / ATR_t)
        """

        price_body = abs(self.close_price - self.open_price)
        price_body_atr = price_body / self._safe_atr
        price_body_atr.name = self.price_feature_columns.PRICE_BODY_ATR

        return self._apply_rounding(price_body_atr)

    @property
    def price_upper_wick_atr(self) -> pd.Series:
        """
        Calculates the upper wick size normalized by ATR:
        (H_t - max(O_t, C_t)) / ATR_t
        """
        upper_wick = self.high_price - np.maximum(self.open_price, self.close_price)
        upper_wick_atr = upper_wick / self._safe_atr

        upper_wick_atr.name = self.price_feature_columns.UPPER_WICK_ATR
        return self._apply_rounding(upper_wick_atr)

    @property
    def price_lower_wick_atr(self) -> pd.Series:
        """
        Calculates the lower wick size normalized by ATR:
        (min(O_t, C_t) - L_t) / ATR_t
        """
        lower_wick = np.minimum(self.open_price, self.close_price) - self.low_price
        lower_wick_atr = lower_wick / self._safe_atr

        lower_wick_atr.name = self.price_feature_columns.LOWER_WICK_ATR
        return self._apply_rounding(lower_wick_atr)

    @property
    def price_close_location(self) -> pd.Series:
        """
        Calculates the candle price location:
        (C_t - L_t)) / (H_t - L_t)
        """
        price_close_location = (self.close_price - self.low_price) / self._price_range
        price_close_location.name = self.price_feature_columns.PRICE_CLOSE_LOCATION
        return self._apply_rounding(price_close_location)

    def price_dist_hh_atr(self, window: int = 9) -> pd.Series:
        """
        Distance of close to rolling highest high, normalized by ATR:
        (C_t - max(H_{t-window:t-1})) / ATR_t
        """
        rolling_hh = (
            self.high_price
            .rolling(window=window, min_periods=window)
            .max()
            .shift(1)
        )

        dist_hh_atr = (self.close_price - rolling_hh) / self._safe_atr
        dist_hh_atr.name = f"{self.price_feature_columns.PRICE_DIST_HH_ATR}_{window}"

        return self._apply_rounding(dist_hh_atr)

    def price_dist_ll_atr(self, window: int = 9) -> pd.Series:
        rolling_ll = (
            self.low_price
            .rolling(window=window, min_periods=window)
            .min()
            .shift(1)
        )

        dist_ll_atr = (self.close_price - rolling_ll) / self._safe_atr
        dist_ll_atr.name = f"{self.price_feature_columns.PRICE_DIST_LL_ATR}_{window}"

        return self._apply_rounding(dist_ll_atr)

    @property
    def _price_range(self) -> pd.Series:
        return self.high_price - self.low_price

    @property
    def _safe_atr(self) -> pd.Series:
        return self.atr.replace(0, np.nan)

    def _apply_rounding(self, series: pd.Series) -> pd.Series:
        return series.round(self.rounding_factor) if self.rounding_factor is not None else series

    def _validate_input(self) -> None:
        """
        Validate that all input Series are aligned and usable.
        """
        indices = {
            "open_price": self.open_price.index,
            "high_price": self.high_price.index,
            "low_price": self.low_price.index,
            "close_price": self.close_price.index,
            "atr": self.atr.index,
        }

        base_name, base_index = next(iter(indices.items()))

        for name, index in indices.items():
            if not index.equals(base_index):
                raise ValueError(
                    f"Index mismatch detected.\n"
                    f"Expected index of '{name}' to match '{base_name}'.\n"
                    f"'{name}' index head: {index[:5]}\n"
                    f"'{base_name}' index head: {base_index[:5]}"
                )

        for name, series in [
            ("open_price", self.open_price),
            ("high_price", self.high_price),
            ("low_price", self.low_price),
            ("close_price", self.close_price),
            ("atr", self.atr),
        ]:
            if not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"{name} must be numeric")





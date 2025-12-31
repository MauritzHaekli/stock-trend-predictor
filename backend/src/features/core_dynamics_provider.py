import pandas as pd
from backend.src.schema.core_dynamics import CoreDynamicsColumns


class CoreDynamicsProvider:
    """
    Computes geometric and dynamic relationships between price and core indicators
    (e.g. EMA distance, slope, acceleration).
    """

    def __init__(
        self,
        close_price: pd.Series,
        ema_series: pd.Series,
        atr_series: pd.Series,
        rounding_factor: int | None = None,
    ):
        self.close = close_price
        self.ema = ema_series
        self.atr = atr_series
        self.rounding_factor = rounding_factor

        self.core_dynamics_columns = CoreDynamicsColumns()

        self._validate_inputs()

    def _validate_inputs(self):
        if not (self.close.index.equals(self.ema.index) and self.close.index.equals(self.atr.index)):
            raise ValueError("Close, EMA, and ATR must share the same index")

    @property
    def core_dynamics(self) -> pd.DataFrame:
        """
        Return all core dynamics features as a single DataFrame.
        """
        core_dynamics = {
            self.core_dynamics_columns.RELATIVE_EMA_DISTANCE: self.ema_relative_distance,
            self.core_dynamics_columns.EMA_DISTANCE_DELTA: self.ema_distance_delta,
            self.core_dynamics_columns.ATR_NORMALIZED_EMA_DISTANCE: self.ema_normalized_atr_distance,
            self.core_dynamics_columns.CLOSE_ABOVE_EMA: self.close_above_ema,
            self.core_dynamics_columns.EMA_RELATIVE_SLOPE: self.ema_relative_slope,
            self.core_dynamics_columns.EMA_ACCELERATION: self.ema_acceleration,
        }

        return pd.DataFrame(core_dynamics, index=self.close.index)

    @property
    def ema_relative_distance(self) -> pd.Series:
        """
        Relative distance of price to EMA:
        (Close - EMA) / EMA
        """
        distance = (self.close - self.ema) / self.ema
        distance.name = self.core_dynamics_columns.RELATIVE_EMA_DISTANCE

        if self.rounding_factor is not None:
            distance = distance.round(self.rounding_factor)

        return distance

    @property
    def ema_distance_delta(self) -> pd.Series:
        delta = self.ema_relative_distance.diff()
        delta.name = self.core_dynamics_columns.EMA_DISTANCE_DELTA
        return delta

    @property
    def ema_normalized_atr_distance(self) -> pd.Series:
        """
        Normalized by ATR distance of price to EMA:
        (Close - EMA) / ATR
        """
        normalized_atr_distance = (self.close - self.ema) / self.atr
        normalized_atr_distance.name = self.core_dynamics_columns.ATR_NORMALIZED_EMA_DISTANCE

        if self.rounding_factor is not None:
            normalized_atr_distance = normalized_atr_distance.round(self.rounding_factor)

        return normalized_atr_distance

    @property
    def close_above_ema(self) -> pd.Series:
        close_above_ema = (self.close > self.ema).astype(int)
        close_above_ema.name = self.core_dynamics_columns.CLOSE_ABOVE_EMA
        return close_above_ema

    @property
    def ema_relative_slope(self) -> pd.Series:
        """
        Relative slope of EMA
        """

        ema_relative_slope = (self.ema - self.ema.shift(1))/self.ema.shift(1)
        ema_relative_slope.name = self.core_dynamics_columns.EMA_RELATIVE_SLOPE

        if self.rounding_factor is not None:
            ema_relative_slope = ema_relative_slope.round(self.rounding_factor)

        return ema_relative_slope

    @property
    def ema_acceleration(self) -> pd.Series:
        acc = self.ema_relative_slope.diff()
        acc.name = self.core_dynamics_columns.EMA_ACCELERATION
        return acc

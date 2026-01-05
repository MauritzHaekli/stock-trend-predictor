import pandas as pd
from backend.src.schema.core_dynamics import CoreDynamicsColumns


class CoreDynamicsProvider:
    """
    Core geometric dynamics between price, EMA, and volatility.

    All features are causal (past-only) and index-aligned.
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
            self.core_dynamics_columns.RELATIVE_EMA_DISTANCE: self.get_ema_relative_distance(distance_periods=0),
            self.core_dynamics_columns.EMA_DISTANCE_CHANGE: self.get_ema_distance_change(periods=1),
            self.core_dynamics_columns.ATR_NORMALIZED_EMA_DISTANCE: self.get_ema_normalized_atr_distance(periods=0),
            self.core_dynamics_columns.EMA_RELATIVE_SLOPE: self.get_ema_relative_slope(periods=1),
            self.core_dynamics_columns.EMA_ACCELERATION: self.get_ema_acceleration(),
            self.core_dynamics_columns.EMA_DISTANCE_STANDARD_DEVIATION : self.get_ema_distance_rolling_std()
        }

        return pd.DataFrame(core_dynamics, index=self.close.index)

    def get_ema_relative_distance(self, distance_periods: int = 0) -> pd.Series:
        """
        Relative distance of price to EMA:
        (Close - EMA) / EMA
        """
        distance = (self.close - self.ema.shift(distance_periods)) / self.ema.shift(distance_periods)
        distance.name = self.core_dynamics_columns.RELATIVE_EMA_DISTANCE

        if self.rounding_factor is not None:
            distance = distance.round(self.rounding_factor)

        return distance

    def get_ema_distance_change(self, periods: int = 1) -> pd.Series:
        """
        Change in relative EMA distance.

        Measures momentum of price relative to EMA.
        Positive values indicate price moving away from EMA.
        Negative values indicate reversion toward EMA.
        """

        ema_distance = self.get_ema_relative_distance(distance_periods=0)
        delta = ema_distance.diff(periods)

        delta.name = self.core_dynamics_columns.EMA_DISTANCE_CHANGE

        if self.rounding_factor is not None:
            delta = delta.round(self.rounding_factor)

        return delta

    def get_ema_normalized_atr_distance(self, periods: int = 0) -> pd.Series:
        """
        Normalized by ATR distance of price to EMA:
        (Close - EMA) / ATR
        """
        normalized_atr_distance = (self.close - self.ema.shift(periods)) / self.atr.shift(periods)
        normalized_atr_distance.name = self.core_dynamics_columns.ATR_NORMALIZED_EMA_DISTANCE

        if self.rounding_factor is not None:
            normalized_atr_distance = normalized_atr_distance.round(self.rounding_factor)

        return normalized_atr_distance


    def get_ema_relative_slope(self, periods:int = 1) -> pd.Series:
        """
        Relative slope of EMA
        """

        ema_relative_slope = (self.ema - self.ema.shift(periods))/self.ema.shift(periods)
        ema_relative_slope.name = self.core_dynamics_columns.EMA_RELATIVE_SLOPE

        if self.rounding_factor is not None:
            ema_relative_slope = ema_relative_slope.round(self.rounding_factor)

        return ema_relative_slope

    def get_ema_acceleration(self) -> pd.Series:
        """
        Second-order difference of EMA (discrete acceleration).
        Positive values indicate increasing slope.
        """
        ema_acceleration = (self.ema - 2 * self.ema.shift(1) + self.ema.shift(2))
        ema_acceleration.name = self.core_dynamics_columns.EMA_ACCELERATION

        if self.rounding_factor is not None:
            ema_acceleration = ema_acceleration.round(self.rounding_factor)
        return ema_acceleration

    def get_ema_distance_rolling_std(self, window: int = 14, normalize_by_atr: bool = False) -> pd.Series:
        """
        Rolling standard deviation of relative EMA distance.

        Measures stability of the price–EMA relationship.
        High values indicate choppy / mean-reverting regimes.
        """

        ema_distance = self.get_ema_relative_distance(distance_periods=0)

        rolling_std = ema_distance.rolling(window=window, min_periods=window).std()

        if normalize_by_atr:
            rolling_std = rolling_std / self.atr

        rolling_std.name = self.core_dynamics_columns.EMA_DISTANCE_STANDARD_DEVIATION

        if self.rounding_factor is not None:
            rolling_std = rolling_std.round(self.rounding_factor)

        return rolling_std


import pandas as pd
import numpy as np
from backend.src.schema.returns import ReturnsColumns


class ReturnsProvider:
    """
    Computes returns of different windows as well as oc return and gap return.
    """

    def __init__(
        self,
        open_price: pd.Series,
        close_price: pd.Series,

        rounding_factor: int | None = None,
    ):
        self.open_price = open_price
        self.close_price = close_price
        self.rounding_factor = rounding_factor

        self.returns_columns = ReturnsColumns()

        self._validate_close_series()

    @property
    def returns(self) -> pd.DataFrame:
        """
        Return all return features as a single DataFrame.
        """
        returns = {
            self.returns_columns.LOG_RETURN_ONE: self.log_return_n(n=1),
            self.returns_columns.LOG_RETURN_TWO: self.log_return_n(n=2),
            self.returns_columns.LOG_RETURN_THREE: self.log_return_n(n=3),
            self.returns_columns.LOG_RETURN_NINE: self.log_return_n(n=9),
            self.returns_columns.OC_LOG_RETURN: self.oc_log_return_n(steps=0),
            self.returns_columns.GAP_LOG_RETURN: self.gap_log_return_n(steps=0)
        }

        return pd.DataFrame(returns, index=self.close_price.index)

    def log_return_n(self, n: int) -> pd.Series:
        """
        Compute n-step log return:
        r_{n,t} = ln(C_t / C_{t-n})

        Parameters
        ----------
        n : int Lookback steps (n >= 1)

        Return: pd.Series n-step log return (NaN for first n rows)
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        log_return = np.log(self.close_price / self.close_price.shift(n))
        log_return.name = f"log_return_{n}"

        if self.rounding_factor is not None:
            log_return = log_return.round(self.rounding_factor)

        return log_return

    def oc_log_return_n(self, steps: int = 0) -> pd.Series:
        oc_log_return = np.log(self.close_price.shift(steps) / self.open_price.shift(steps))
        oc_log_return.name = self.returns_columns.OC_LOG_RETURN

        if self.rounding_factor is not None:
            oc_log_return = oc_log_return.round(self.rounding_factor)

        return oc_log_return

    def gap_log_return_n(self, steps: int = 0) -> pd.Series:
        """
        gap_log_return(0) = ln(O_t / C_{t-1})
        gap_log_return(1) = ln(O_{t-1} / C_{t-2})
        """
        gap_log_return = np.log(self.open_price.shift(steps) / self.close_price.shift(steps + 1))
        gap_log_return.name = self.returns_columns.GAP_LOG_RETURN

        if self.rounding_factor is not None:
            gap_log_return = gap_log_return.round(self.rounding_factor)

        return gap_log_return

    def _validate_close_series(self) -> None:
        """
        Validates correctness of close price time series.
        """
        if (self.open_price <= 0).any():
            raise ValueError("Open prices must be strictly positive")

        if (self.close_price <= 0).any():
            raise ValueError("Close prices must be strictly positive")

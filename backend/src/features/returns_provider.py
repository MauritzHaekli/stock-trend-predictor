import pandas as pd
import numpy as np
from backend.src.schema.returns import ReturnsColumns


class ReturnsProvider:
    """
    Computes geometric and dynamic relationships between price and core indicators
    (e.g. EMA distance, slope, acceleration).
    """

    def __init__(
        self,
        close_price: pd.Series,
        rounding_factor: int | None = None,
    ):
        self.close = close_price
        self.rounding_factor = rounding_factor

        self.returns_columns = ReturnsColumns()


    @property
    def returns(self) -> pd.DataFrame:
        """
        Return all return features as a single DataFrame.
        """
        returns = {
            self.returns_columns.LOG_RETURN_ONE: self.log_return_n(n=1),
            self.returns_columns.LOG_RETURN_TWO: self.log_return_n(n=2),
            self.returns_columns.LOG_RETURN_THREE: self.log_return_n(n=3),
            self.returns_columns.LOG_RETURN_NINE: self.log_return_n(n=9)

        }

        return pd.DataFrame(returns, index=self.close.index)

    def log_return_n(self, n: int,rounding_factor: int | None = None,
    ) -> pd.Series:
        """
        Compute n-step log return:
        r_{n,t} = ln(C_t / C_{t-n})

        Parameters
        ----------
        n : int Lookback steps (n >= 1)
        rounding_factor : int | None Optional rounding

        Return: pd.Series n-step log return (NaN for first n rows)
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        if (self.close <= 0).any():
            raise ValueError("Close prices must be strictly positive")

        log_return = np.log(self.close).diff(n)
        log_return.name = f"log_return_{n}"

        if rounding_factor is not None:
            log_return = log_return.round(rounding_factor)

        return log_return

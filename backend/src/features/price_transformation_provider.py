import pandas as pd
import numpy as np
from backend.src.schema.ohlcv import OHLCVColumns
from backend.src.schema.price_transformation import PriceTransformationColumns


class PriceTransformationProvider:
    """
    Price transformations
    """

    def __init__(
        self,
        time_series: pd.DataFrame,
        rounding_factor: int | None = None,
    ):
        self.time_series = time_series

        self.rounding_factor = rounding_factor


        self.OHLCVColumns = OHLCVColumns()
        self.price_transformation_columns = PriceTransformationColumns()


    @property
    def price_transformations(self) -> pd.DataFrame:
        """
        Return all price transformation features as a single DataFrame.
        """
        price_transformations = {
            self.price_transformation_columns.LOG_OPEN: self.get_log_series(self.time_series[self.OHLCVColumns.OPEN]),
            self.price_transformation_columns.LOG_HIGH: self.get_log_series(self.time_series[self.OHLCVColumns.HIGH]),
            self.price_transformation_columns.LOG_LOW: self.get_log_series(self.time_series[self.OHLCVColumns.LOW]),
            self.price_transformation_columns.LOG_CLOSE: self.get_log_series(self.time_series[self.OHLCVColumns.CLOSE])
        }

        return pd.DataFrame(price_transformations, index=self.time_series.index)

    def get_log_series(self, series: pd.Series) -> pd.Series:
        log_series: pd.Series = np.log(series)
        if self.rounding_factor is not None:
            log_series = log_series.round(self.rounding_factor)
        return  log_series
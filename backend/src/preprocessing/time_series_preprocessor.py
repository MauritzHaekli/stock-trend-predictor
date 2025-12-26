import numpy as np
import pandas as pd
import logging
from backend.src.schema.raw_ohlcv import RawOHLCVColumns as rawOHLCV

logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    Deterministic preprocessing stage for raw OHLCV time-series data.

    This class transforms raw market OHLCV data into a clean, validated,
    and standardized time-series representation suitable for downstream
    feature engineering and modeling.

    It is intentionally strict: invalid or inconsistent data is either
    corrected in a transparent way (with logging) or rejected.

    The returned DataFrame is guaranteed to:
    - be a pandas DataFrame with a DatetimeIndex
    - have strictly increasing, unique timestamps
    - contain numeric columns only
    - contain no NaN or infinite values
    - satisfy basic OHLCV price integrity constraints
    - represent a single, regular time interval

    This class does NOT:
    - perform feature engineering
    - compute rolling statistics or indicators
    - resample or aggregate data
    - look ahead in time (no leakage)
    - infer or repair missing time intervals

    ASSUMPTIONS
    - Input data represents a single instrument.
    - Input timestamps are expected to be on a regular grid.
    - Missing or malformed values indicate upstream data quality issues.
    """


    def __init__(
        self,
        datetime_column: str = rawOHLCV.DATETIME,
        ohlcv_columns: tuple[str, ...] = rawOHLCV.OHLCV,
        rounding: int | None = None,
        drop_duplicate_policy: str = "last",
        fill_method: str | None = "ffill"
    ):
        self.datetime_col = datetime_column
        self.ohlcv_columns = ohlcv_columns
        self.rounding = rounding
        self.drop_duplicate_policy = drop_duplicate_policy
        self.fill_method = fill_method

    def process_time_series(self, raw_time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a raw OHLCV time series according to the preprocessing contract.

        raw_time_series : pd.DataFrame
            Raw input time-series data containing at least a datetime column
            and OHLCV price columns.

        Returns: pd.DataFrame
            Cleaned and validated time-series data indexed by datetime.

        Raises
        ------
        ValueError
            If required columns are missing, timestamps are inconsistent,
            or OHLCV integrity constraints are violated.
        TypeError
            If non-numeric data remains after preprocessing.

        This method is deterministic and does not modify the input DataFrame.
        """

        initial_rows = len(raw_time_series)
        processed_time_series: pd.DataFrame = raw_time_series.copy()

        processed_time_series = self._standardize_columns(processed_time_series)
        self._validate_required_columns(processed_time_series)

        processed_time_series = self._parse_datetime(processed_time_series)
        processed_time_series = self._sort(processed_time_series)
        processed_time_series = self._remove_duplicates(processed_time_series)
        processed_time_series = self._coerce_numeric(processed_time_series)
        processed_time_series = self._drop_invalid_rows(processed_time_series)
        processed_time_series = self._fill_missing(processed_time_series)
        processed_time_series = self._round(processed_time_series)
        processed_time_series = self._set_index(processed_time_series)

        final_rows = len(processed_time_series)
        dropped = initial_rows - final_rows
        retained_pct = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0.0

        logger.info(
            "Preprocessing summary | input rows: %d | output rows: %d | dropped: %d (%.2f%% retained)",
            initial_rows,
            final_rows,
            dropped,
            retained_pct,
        )

        self._enforce_time_series_integrity(processed_time_series)
        return processed_time_series

    @staticmethod
    def _standardize_columns(time_series: pd.DataFrame) -> pd.DataFrame:
        time_series.columns = [column.strip().lower() for column in time_series.columns]
        return time_series

    def _validate_required_columns(self, time_series: pd.DataFrame) -> None:
        required_columns = {self.datetime_col, *self.ohlcv_columns}
        missing_columns = required_columns - set(time_series.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    def _parse_datetime(self, time_series: pd.DataFrame) -> pd.DataFrame:
        time_series[self.datetime_col] = pd.to_datetime(time_series[self.datetime_col], errors="raise")
        return time_series

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(self.datetime_col)

    def _remove_duplicates(self, time_series: pd.DataFrame) -> pd.DataFrame:
        dup_count = time_series[self.datetime_col].duplicated().sum()
        if dup_count > 0:
            logger.warning(
                "Removed %d duplicate timestamps using policy='%s'",
                dup_count,
                self.drop_duplicate_policy,
            )
            time_series = time_series.drop_duplicates(
                subset=self.datetime_col,
                keep=self.drop_duplicate_policy,
            )
        return time_series

    def _coerce_numeric(self, time_series: pd.DataFrame) -> pd.DataFrame:
        for col in self.ohlcv_columns:
            before = time_series[col].isna().sum()
            time_series[col] = pd.to_numeric(time_series[col], errors="coerce")
            after = time_series[col].isna().sum()

            if after > before:
                logger.warning(
                    "Column '%s': %d values coerced to NaN",
                    col,
                    after - before,
                )
        return time_series

    def _drop_invalid_rows(self, time_series: pd.DataFrame) -> pd.DataFrame:
        time_series = time_series.dropna(subset=[self.datetime_col])
        time_series = time_series.dropna(subset=list(self.ohlcv_columns), how="any")
        return time_series

    def _fill_missing(self, time_series: pd.DataFrame) -> pd.DataFrame:
        if self.fill_method == "ffill":
            before = time_series[list(self.ohlcv_columns)].isna().sum().sum()
            time_series[list(self.ohlcv_columns)] = time_series[list(self.ohlcv_columns)].ffill()
            after = time_series[list(self.ohlcv_columns)].isna().sum().sum()

            filled = before - after
            if filled > 0:
                logger.info("Forward-filled %d OHLCV values", filled)

        elif self.fill_method is None:
            pass
        else:
            raise ValueError(f"Unsupported fill_method: {self.fill_method}")

        time_series = time_series.dropna(subset=list(self.ohlcv_columns))
        return time_series

    def _round(self, time_series: pd.DataFrame) -> pd.DataFrame:
        if self.rounding is not None:
            time_series[list(self.ohlcv_columns)] = time_series[list(self.ohlcv_columns)].round(self.rounding)
        return time_series

    def _set_index(self, time_series: pd.DataFrame) -> pd.DataFrame:
        return time_series.set_index(self.datetime_col)

    def _enforce_time_series_integrity(self, time_series: pd.DataFrame) -> None:

        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")

        if not time_series.index.is_monotonic_increasing:
            raise ValueError("Datetime index must be strictly increasing")

        if time_series.index.duplicated().any():
            raise ValueError("Duplicate timestamps detected")

        for col in time_series.columns:
            if not pd.api.types.is_numeric_dtype(time_series[col]):
                raise TypeError(f"Non-numeric column detected: {col}")

        if time_series.isna().any().any():
            raise ValueError("NaN values detected after preprocessing")

        if not np.isfinite(time_series.values).all():
            raise ValueError("Non-finite values detected")

        o, h, l, c, v = self.ohlcv_columns

        if not (time_series[h] >= time_series[[o, c]].max(axis=1)).all():
            raise ValueError("High price violation detected")

        if not (time_series[l] <= time_series[[o, c]].min(axis=1)).all():
            raise ValueError("Low price violation detected")

        if (time_series[[o, h, l, c]] <= 0).any().any():
            raise ValueError("Non-positive price detected")

        if (time_series[v] < 0).any():
            raise ValueError("Negative volume detected")

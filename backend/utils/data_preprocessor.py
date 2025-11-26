import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from backend.utils.feature_change_calculator import FeatureChangeCalculator


class DataPreprocessor:
    def __init__(
        self,
        time_series: pd.DataFrame,
        lookback_period: int,
        target_feature: str,
        trend_length: int,
        scaler_path: str = "scaler.save",
        fit_scaler: bool = True
    ):
        self.lookback_period = lookback_period
        self.target = target_feature
        self.trend_length = trend_length
        self.label_column = "label"
        self.target_trend_column = "target trend"
        self.scaler_path = scaler_path
        self.fit_scaler = fit_scaler

        self.time_series = time_series
        self.target_data = self.get_target_data(self.time_series)

        self.feature_data = self.get_feature_data(self.target_data)
        self.label_data = self.get_lookback_labels(self.target_data)

        if self.fit_scaler:
            self.feature_data_scaled = self.fit_scaler_and_transform(self.feature_data)
        else:
            self.feature_data_scaled = self.load_scaler_and_transform(self.feature_data)

        self.feature_data_batched = self.get_feature_data_batches(self.feature_data_scaled)

        self.feature_data_batched = self.feature_data_batched[:-self.trend_length]
        self.label_data = self.label_data[:-self.trend_length]

    def get_target_data(self, time_series: pd.DataFrame) -> pd.DataFrame:
        datetime_col = "datetime"

        if datetime_col not in time_series.columns:
            raise ValueError(f"Expected '{datetime_col}' column in input data.")
        if self.target not in time_series.columns:
            raise ValueError(f"Expected target column '{self.target}' in input data.")

        target_data = time_series.copy()

        target_data[self.target_trend_column] = self.get_current_trend(target_data[self.target])
        target_data[self.label_column] = self.get_future_trend(target_data[self.target], periods=self.trend_length)

        try:
            target_data.index = pd.to_datetime(target_data[datetime_col], format="%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise ValueError(f"Failed to convert '{datetime_col}' to datetime: {e}")

        target_data.drop(columns=[datetime_col], inplace=True, errors='ignore')
        target_data = target_data.iloc[:-self.trend_length]

        return target_data

    def get_current_trend(self, feature: pd.Series) -> pd.Series:
        """
        Current trend: 1 if current value >= previous value, else 0.
        """
        trend = (feature >= feature.shift(1)).astype(int)
        trend.name = self.target_trend_column
        return trend

    def get_future_trend(self, feature: pd.Series, periods: int) -> pd.Series:
        """
        Future trend: 1 if future value >= current value, else 0.
        """
        trend = (feature.shift(-periods) >= feature).astype(int)
        trend.name = self.label_column
        return trend


    def get_feature_data(self, target_data: pd.DataFrame) -> pd.DataFrame:
        if self.label_column not in target_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in the input data.")
        return target_data.drop(columns=[self.label_column, self.target_trend_column])

    def get_lookback_labels(self, data: pd.DataFrame) -> np.ndarray:
        if len(data) <= self.lookback_period:
            raise ValueError("Input data must have more rows than the lookback period.")
        if self.label_column not in data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in input data.")
        labels = data[self.label_column].values
        return labels[self.lookback_period:]

    def fit_scaler_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(df.values)
        joblib.dump(scaler, self.scaler_path)
        return pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    def load_scaler_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = joblib.load(self.scaler_path)
        scaled_values = scaler.transform(df.values)
        return pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    def get_feature_data_batches(self, data: pd.DataFrame) -> np.ndarray:
        if len(data) <= self.lookback_period:
            raise ValueError("Input data must have more rows than the lookback period.")

        values = data.values
        num_batches = len(data) - self.lookback_period
        n_features = values.shape[1]

        batches = np.empty((num_batches, self.lookback_period, n_features), dtype=values.dtype)
        for i in range(num_batches):
            batches[i] = values[i : i + self.lookback_period]

        return batches

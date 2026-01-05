import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class FeatureScaler:
    """
    Scales feature data using a fitted sklearn scaler (e.g. MinMaxScaler).

    RESPONSIBILITIES
    ----------------
    - Fit scaler on training features
    - Transform features consistently across train/val/test/inference
    - Persist and reload the scaler

    GUARANTEES
    ----------
    - No label leakage (labels are never passed here)
    - Column order is preserved
    - Index is preserved
    """

    def __init__(
        self,
        scaler_path: str | Path,
        feature_range: tuple[float, float] = (0.0, 1.0)
    ):
        self.scaler_path = Path(scaler_path)
        self.feature_range = feature_range
        self.scaler: MinMaxScaler | None = None

    def fit_and_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler on training features and transform them.

        This method MUST be called only on training data.
        """

        self._validate_features(features)

        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        scaled_values = self.scaler.fit_transform(features.values)

        self._save_scaler()

        return self._to_dataframe(features, scaled_values)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using an already-fitted scaler.
        """

        self._validate_features(features)

        if self.scaler is None:
            self._load_scaler()

        scaled_values = self.scaler.transform(features.values)

        return self._to_dataframe(features, scaled_values)

    def _save_scaler(self) -> None:
        joblib.dump(self.scaler, self.scaler_path)

    def _load_scaler(self) -> None:
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {self.scaler_path}. "
                "Call fit_and_transform() first."
            )
        self.scaler = joblib.load(self.scaler_path)

    @staticmethod
    def _to_dataframe(
        original: pd.DataFrame,
        scaled_values: np.ndarray
    ) -> pd.DataFrame:
        return pd.DataFrame(
            scaled_values,
            columns=original.columns,
            index=original.index
        )

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")

        if features.empty:
            raise ValueError("features DataFrame is empty")

        if not all(np.issubdtype(dtype, np.number) for dtype in features.dtypes):
            raise TypeError("All feature columns must be numeric")

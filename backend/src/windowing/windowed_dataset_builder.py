import numpy as np
import pandas as pd


class WindowedDatasetBuilder:
    """
    Builds sliding-window datasets for sequence models (e.g. LSTM).

    ASSUMPTIONS
    -----------
    - Features are preprocessed, feature-engineered, and scaled
    - Labels are generated and row-aligned with features
    - Features are time-ordered
    - No splitting happens here (Splitting of data is handled upstream via different stock selection for train/val/test)
    """

    def __init__(self, window_size: int):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size

    def build_sliding_window_dataset(self, features: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Each output sample windowed_feature_sequences[i] consists of features from
        [t - window_size + 1, ..., t] and is paired with label y[t].
        """

        self._validate_inputs(features, labels)

        windowed_sequences: np.ndarray = self._build_sliding_windows(features.values)
        window_adjusted_labels: np.ndarray = labels[self.window_size - 1:]

        if len(windowed_sequences) != len(window_adjusted_labels):
            raise RuntimeError(
                f"Window/label mismatch: windows={windowed_sequences.shape}, "
                f"labels={window_adjusted_labels.shape}"
            )

        return windowed_sequences, window_adjusted_labels

    def _build_sliding_windows(self, values: np.ndarray) -> np.ndarray:
        """
        Convert 2D feature array into overlapping sliding windows.
        """
        n_rows, n_features = values.shape
        n_samples = n_rows - self.window_size + 1

        sliding_windows: np.ndarray = np.empty((n_samples, self.window_size, n_features), dtype=np.float32)

        for i in range(n_samples):
            sliding_windows[i] = values[i : i + self.window_size]

        return sliding_windows

    def _validate_inputs(self, features: pd.DataFrame, labels: np.ndarray) -> None:

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")

        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a NumPy array")

        if not features.index.is_monotonic_increasing or features.index.has_duplicates:
            raise ValueError("Feature index must be strictly increasing")

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")

        if len(features) != len(labels):
            raise ValueError(
                f"Feature/label length mismatch: "
                f"{len(features)} != {len(labels)}"
            )

        if len(features) < self.window_size:
            raise ValueError(
                "Not enough rows to build a single window"
            )

import logging

import numpy as np
import pandas as pd
from backend.src.preprocessing.time_series_preprocessor import TimeSeriesPreprocessor
from backend.src.features.feature_provider import FeatureProvider
from backend.src.labeling.label_generator import LabelGenerator
from backend.src.features.feature_scaler import FeatureScaler
from backend.src.windowing.windowed_dataset_builder import WindowedDatasetBuilder
from backend.src.models.stock_trend_lstm.stock_trend_lstm_model import StockTrendLSTMModel

logger = logging.getLogger(__name__)


class StockTrendLSTMPipeline:
    """
    End-to-end pipeline for LSTM-based stock trend prediction.

    PIPELINE STAGES
    ----------------
    1. Time-series preprocessing
    2. Feature generation
    3. Label generation
    4. Feature scaling
    5. Sliding-window dataset construction
    6. Model training

    DESIGN PRINCIPLES
    -----------------
    - No work is done in __init__ (lazy execution)
    - Each stage is explicit and debuggable
    - Train/validation split handled upstream (e.g. different stocks)
    """

    def __init__(
        self,
        training_time_series: pd.DataFrame,
        validation_time_series: pd.DataFrame,
        target_feature: str,
        trend_length: int,
        window_size: int,
        epochs: int,
        batch_size: int,
        scaler_save_path: str,
    ):
        self.training_data = training_time_series
        self.validation_data = validation_time_series

        self.target_feature = target_feature
        self.trend_length = trend_length
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_save_path = scaler_save_path

        self.labeled_training_ts: pd.DataFrame | None = None
        self.labeled_validation_ts: pd.DataFrame | None = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.model_wrapper: StockTrendLSTMModel | None = None
        self.history = None

        logger.info("StockTrendLSTMPipeline initialized")

    def prepare_data(self) -> None:
        logger.info("Starting data preparation")

        preprocessor: TimeSeriesPreprocessor = TimeSeriesPreprocessor()
        train_preprocessed: pd.DataFrame = preprocessor.process_time_series(self.training_data)
        val_preprocessed: pd.DataFrame = preprocessor.process_time_series(self.validation_data)

        train_features_ts: pd.DataFrame = FeatureProvider(train_preprocessed).feature_time_series
        val_features_ts: pd.DataFrame = FeatureProvider(val_preprocessed).feature_time_series

        label_generator: LabelGenerator = LabelGenerator(
            target_feature=self.target_feature,
            trend_length=self.trend_length,
            window_size=self.window_size,
        )

        labeled_train: pd.DataFrame = label_generator.get_labeled_feature_data(train_features_ts)
        labeled_val: pd.DataFrame = label_generator.get_labeled_feature_data(val_features_ts)

        self.labeled_training_ts = labeled_train
        self.labeled_validation_ts = labeled_val

        train_features: pd.DataFrame = label_generator.get_aligned_features(labeled_train)
        val_features: pd.DataFrame = label_generator.get_aligned_features(labeled_val)

        train_labels: np.ndarray = label_generator.get_window_aligned_labels(labeled_train)
        val_labels: np.ndarray = label_generator.get_window_aligned_labels(labeled_val)

        scaler: FeatureScaler = FeatureScaler(self.scaler_save_path)
        scaled_train_features: pd.DataFrame = scaler.fit_and_transform(train_features)
        scaled_val_features: pd.DataFrame = scaler.transform(val_features)

        dataset_builder: WindowedDatasetBuilder = WindowedDatasetBuilder(self.window_size)
        self.X_train, self.y_train = dataset_builder.build_sliding_window_dataset(
            scaled_train_features, train_labels
        )
        self.X_val, self.y_val = dataset_builder.build_sliding_window_dataset(
            scaled_val_features, val_labels
        )

        self._log_dataset_summary()
        logger.info("Data preparation completed")

    def build_model(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data must be prepared before building the model")

        self.model_wrapper = StockTrendLSTMModel(
            batched_training_data=self.X_train,
            batched_validation_data=self.X_val,
            label_training_data=self.y_train,
            label_validation_data=self.y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

        logger.info("Model built successfully")

    def train(self):
        if self.model_wrapper is None:
            raise RuntimeError("Model must be built before training")

        logger.info("Starting model training")
        self.history = self.model_wrapper.train_model()
        logger.info("Training completed")

        return self.history

    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_model(self) -> StockTrendLSTMModel:
        if self.model_wrapper is None:
            raise RuntimeError("Model has not been built yet")
        return self.model_wrapper

    def _log_dataset_summary(self) -> None:
        logger.info(f"X_train shape: {self.X_train.shape}")
        logger.info(f"y_train mean: {self.y_train.mean():.4f}")
        logger.info(f"X_val shape: {self.X_val.shape}")
        logger.info(f"y_val mean: {self.y_val.mean():.4f}")


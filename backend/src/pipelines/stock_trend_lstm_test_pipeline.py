import logging
import numpy as np
import pandas as pd
from backend.src.preprocessing.time_series_preprocessor import TimeSeriesPreprocessor
from backend.src.features.feature_provider import FeatureProvider
from backend.src.labeling.label_generator import LabelGenerator
from backend.src.scaling.feature_scaler import FeatureScaler
from backend.src.windowing.windowed_dataset_builder import WindowedDatasetBuilder
from backend.src.utils.config import Config

config = Config("C:/Users/mohae/Desktop/StockTrendPredictor/backend/config.yaml")

logger = logging.getLogger(__name__)


class StockTrendLSTMTestPipeline:
    """
    Testing pipeline for LSTM-based stock trend prediction.
    """

    def __init__(
        self,
        testing_time_series: pd.DataFrame,
        target_feature: str,
        trend_length: int,
        window_size: int,
        scaler_path: str,
    ):
        self.testing_time_series = testing_time_series

        self.target_feature = target_feature
        self.trend_length = trend_length
        self.window_size = window_size

        self.scaler_path = scaler_path

        self.labeled_testing_ts: pd.DataFrame | None = None

        self.X_test = None
        self.y_test = None

        logger.info("StockTrendLSTMTestPipeline initialized")

    def prepare_data(self) -> None:
        logger.info("Starting data preparation")

        preprocessor: TimeSeriesPreprocessor = TimeSeriesPreprocessor()
        test_preprocessed: pd.DataFrame = preprocessor.process_time_series(self.testing_time_series)

        test_features_ts: pd.DataFrame = FeatureProvider(test_preprocessed, config.as_dict()).feature_time_series

        label_generator: LabelGenerator = LabelGenerator(
            target_feature=self.target_feature,
            trend_length=self.trend_length,
            window_size=self.window_size,
        )

        labeled_test: pd.DataFrame = label_generator.get_labeled_feature_data(test_features_ts)

        self.labeled_testing_ts = labeled_test

        test_features: pd.DataFrame = label_generator.get_aligned_features(labeled_test)
        test_labels: np.ndarray = label_generator.get_window_aligned_labels(labeled_test)

        scaler: FeatureScaler = FeatureScaler(self.scaler_path)
        scaled_test_features: pd.DataFrame = scaler.transform(test_features)

        dataset_builder: WindowedDatasetBuilder = WindowedDatasetBuilder(self.window_size)

        self.X_test, self.y_test = dataset_builder.build_sliding_window_dataset(
            scaled_test_features, test_labels
        )

        logger.info("Test data preparation completed")

    def get_test_data(self):
        return self.X_test, self.y_test
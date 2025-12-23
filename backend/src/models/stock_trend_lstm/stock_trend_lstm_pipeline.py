import joblib
import pandas as pd
import logging
from backend.src.models.time_series_preprocessor import TimeSeriesPreprocessor
from backend.src.models.feature_provider import FeatureProvider
from backend.src.models.label_generator import LabelGenerator
from backend.src.models.feature_scaler import FeatureScaler
from backend.src.models.windowed_dataset_builder import WindowedDatasetBuilder
from backend.src.models.stock_trend_lstm.stock_trend_lstm_model import StockTrendLSTMModel


logger = logging.getLogger(__name__)

class StockTrendLSTMPipeline:
    def __init__(self,
                 training_time_series: pd.DataFrame,
                 validation_time_series: pd.DataFrame,
                 target_feature: str,
                 trend_length: int,
                 window_size: int,
                 epochs: int,
                 batch_size: int):

        self.training_data: pd.DataFrame = training_time_series
        self.validation_data: pd.DataFrame = validation_time_series
        self.target_feature: str = target_feature
        self.trend_length: int = trend_length
        self.window_size: int = window_size
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        self.label_column: str = "label"

        self.scaler_save_path: str = f"C:/Users/mohae/Desktop/StockTrendPredictor/backend/src/models/saved scalers/scaler.save"

        logger.info("StockTrendLSTMPipeline initiated")

        time_series_preprocessor: TimeSeriesPreprocessor = TimeSeriesPreprocessor()

        training_feature_provider = FeatureProvider(self.training_data)
        training_feature_time_series = training_feature_provider.feature_time_series

        validation_feature_provider = FeatureProvider(self.validation_data)
        validation_feature_time_series = validation_feature_provider.feature_time_series

        preprocessed_training_data: pd.DataFrame = time_series_preprocessor.process_time_series(training_feature_time_series)
        preprocessed_validation_data: pd.DataFrame = time_series_preprocessor.process_time_series(validation_feature_time_series)

        logger.info("Preprocessing training/validation data completed")

        label_generator: LabelGenerator = LabelGenerator(self.target_feature, self.trend_length, self.window_size)

        self.labeled_train: pd.DataFrame = label_generator.get_labeled_feature_data(preprocessed_training_data)
        self.labeled_valid: pd.DataFrame = label_generator.get_labeled_feature_data(preprocessed_validation_data)

        feature_scaler: FeatureScaler = FeatureScaler(self.scaler_save_path)

        self.train_labels_full = label_generator.get_window_aligned_labels(self.labeled_train)
        self.valid_labels_full = label_generator.get_window_aligned_labels(self.labeled_valid)

        self.train_features = label_generator.get_aligned_features(self.labeled_train)
        self.valid_features = label_generator.get_aligned_features(self.labeled_valid)

        self.scaled_training_feature_data = feature_scaler.fit_and_transform(self.train_features)
        self.scaled_validation_feature_data = feature_scaler.transform(self.valid_features)

        dataset_builder: WindowedDatasetBuilder = WindowedDatasetBuilder(self.window_size)

        self.X_train, self.y_train = dataset_builder.build_sliding_window_dataset(self.scaled_training_feature_data, self.train_labels_full)

        self.X_val, self.y_val = dataset_builder.build_sliding_window_dataset(self.scaled_validation_feature_data, self.valid_labels_full)

        stock_trend_lstm_model: StockTrendLSTMModel = StockTrendLSTMModel(self.X_train, self.X_val, self.y_train, self.y_val, self.epochs, self.batch_size)

        self.history = stock_trend_lstm_model.train_model()

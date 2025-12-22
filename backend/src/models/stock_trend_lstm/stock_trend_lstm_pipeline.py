import joblib
import numpy as np
import pandas as pd
import logging
from backend.src.models.time_series_preprocessor import TimeSeriesPreprocessor
from backend.src.models.label_generator import LabelGenerator
from backend.src.models.windowed_dataset_builder import WindowedDatasetBuilder
from backend.src.models.stock_trend_lstm.stock_trend_lstm_model import StockTrendLSTMModel
from sklearn.preprocessing import MinMaxScaler


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
        label_generator: LabelGenerator = LabelGenerator(self.target_feature, self.trend_length, self.window_size)

        preprocessed_training_data: pd.DataFrame = time_series_preprocessor.process_time_series(self.training_data)
        preprocessed_validation_data: pd.DataFrame = time_series_preprocessor.process_time_series(self.validation_data)

        logger.info("Preprocessing training/validation data completed")

        # 1) Get labeled data (labels + trimmed tail)
        labeled_train: pd.DataFrame = label_generator.get_labeled_feature_data(preprocessed_training_data)
        labeled_valid: pd.DataFrame = label_generator.get_labeled_feature_data(preprocessed_validation_data)

        # 2) Extract labels as arrays (NO alignment here yet)
        train_labels_full = labeled_train[self.label_column].values
        valid_labels_full = labeled_valid[self.label_column].values

        # 3) Remove label column from features (keep only numeric features for scaling)
        train_features = labeled_train.drop(columns=[self.label_column])
        valid_features = labeled_valid.drop(columns=[self.label_column])

        self.scaled_training_feature_data = self.fit_scaler_and_transform(train_features)
        self.scaled_validation_feature_data = self.load_scaler_and_transform(valid_features)

        dataset_builder: WindowedDatasetBuilder = WindowedDatasetBuilder(self.window_size)

        self.X_train, self.y_train = dataset_builder.build_sliding_window_dataset(self.scaled_training_feature_data, train_labels_full)

        self.X_val, self.y_val = dataset_builder.build_sliding_window_dataset(self.scaled_validation_feature_data, valid_labels_full)

        stock_trend_lstm_model: StockTrendLSTMModel = StockTrendLSTMModel(self.X_train,
                                                                          self.X_val,
                                                                          self.y_train,
                                                                          self.y_val,
                                                                          self.epochs,
                                                                          self.batch_size)

        self.history = stock_trend_lstm_model.train_model()


    def fit_scaler_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit MinMaxScaler on training features and transform them.
        Assumes df has NO label column already.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(df.values)
        joblib.dump(scaler, self.scaler_save_path)
        return pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    def load_scaler_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load the previously fitted scaler and transform validation features.
        """
        scaler = joblib.load(self.scaler_save_path)
        scaled_values = scaler.transform(df.values)
        return pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

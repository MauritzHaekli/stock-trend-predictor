import joblib
import numpy as np
import pandas as pd
import logging
from backend.src.models.stock_trend_lstm.time_series_preprocessor import TimeSeriesPreprocessor
from backend.src.models.stock_trend_lstm.label_generator import LabelGenerator
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
        labeled_train = label_generator.get_labeled_feature_data(preprocessed_training_data)
        labeled_valid = label_generator.get_labeled_feature_data(preprocessed_validation_data)

        # 2) Extract labels as arrays (NO alignment here yet)
        train_labels_full = labeled_train[self.label_column].values
        valid_labels_full = labeled_valid[self.label_column].values

        # 3) Remove label column from features (keep only numeric features for scaling)
        train_features = labeled_train.drop(columns=[self.label_column])
        valid_features = labeled_valid.drop(columns=[self.label_column])

        self.scaled_training_feature_data = self.fit_scaler_and_transform(train_features)
        self.scaled_validation_feature_data = self.load_scaler_and_transform(valid_features)

        self.batched_training_feature_data = self.get_feature_data_batches(self.scaled_training_feature_data)
        self.batched_validation_feature_data = self.get_feature_data_batches(self.scaled_validation_feature_data)

        # Align labels to sequence windows:
        # For each window [t ... t+window_size-1], we use label at t+window_size-1
        self.label_training_data = train_labels_full[self.window_size - 1:]
        self.label_validation_data = valid_labels_full[self.window_size - 1:]

        stock_trend_lstm_model: StockTrendLSTMModel = StockTrendLSTMModel(self.batched_training_feature_data,
                                                                          self.batched_validation_feature_data,
                                                                          self.label_training_data,
                                                                          self.label_validation_data,
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

    def get_feature_data_batches(self, data: pd.DataFrame) -> np.ndarray:
        """
        Build 3D tensor of shape (num_samples, window_size, num_features)
        using sliding windows over the scaled feature dataframe.
        """
        if len(data) <= self.window_size:
            raise ValueError("Input data must have more rows than the window size.")

        values = data.values
        n_samples = len(data) - self.window_size + 1  # +1 so last window ends at last row
        n_features = values.shape[1]

        batches = np.empty((n_samples, self.window_size, n_features), dtype=values.dtype)
        for i in range(n_samples):
            batches[i] = values[i: i + self.window_size]

        return batches
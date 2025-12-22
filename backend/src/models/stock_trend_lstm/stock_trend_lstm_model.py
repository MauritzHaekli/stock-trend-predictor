import logging
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import Accuracy,Precision, Recall, AUC
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History

logger = logging.getLogger(__name__)

class StockTrendLSTMModel:
    """
        LSTM-based binary classification model for stock trend prediction.

        This class assumes:
        - Input features are already scaled and windowed
        - Labels are aligned with feature windows
        - No preprocessing, scaling, or labeling occurs here

        Responsibilities:
        - Validate batched input data
        - Build and compile the LSTM model
        - Train with early stopping
        - Persist the trained model

        This class is intentionally stateless with respect to data preparation
        and should be used as the final stage of the ML pipeline
        """

    def __init__(self,
                 batched_training_data: np.ndarray,
                 batched_validation_data: np.ndarray,
                 label_training_data: np.ndarray,
                 label_validation_data: np.ndarray,
                 epochs: int,
                 batch_size: int):

        self.batched_training_data: np.ndarray = batched_training_data
        self.batched_validation_data: np.ndarray = batched_validation_data
        self.label_training_data: np.ndarray = label_training_data
        self.label_validation_data: np.ndarray = label_validation_data
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        self.validate_data_input()

        self.model: Model | None = None
        self.save_name: str = "StockTrendLSTM"
        self.model_save_path: str = f"C:/Users/mohae/Desktop/StockTrendPredictor/backend/src/models/saved models/trained_{self.save_name}_model.keras"
        self.input_shape: tuple[int, int] = self.get_input_shape(self.batched_training_data)

        self.build_model()

    @staticmethod
    def get_input_shape(batched_training_data: np.ndarray) -> tuple[int, int]:
        """
        Infer (timesteps, num_features) from batched training data.
        """
        if batched_training_data.ndim != 3:
            raise ValueError(f"Expected 3D data for LSTM, got shape {batched_training_data.shape}")
        timesteps: int = batched_training_data.shape[1]
        n_features: int = batched_training_data.shape[2]
        return timesteps, n_features

    def validate_data_input(self):
        # Sanity check: X and y lengths must match
        if len(self.batched_training_data) != len(self.label_training_data):
            raise ValueError(
                f"Training X/y length mismatch: X={len(self.batched_training_data)}, "
                f"y={len(self.label_training_data)}"
            )
        if len(self.batched_validation_data) != len(self.label_validation_data):
            raise ValueError(
                f"Validation X/y length mismatch: X={len(self.batched_validation_data)}, "
                f"y={len(self.label_validation_data)}"
            )


    def build_model(self) -> None:
        """
        Build the LSTM model with the inferred input shape.
        """
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=self.input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))

        learning_rate: float = 0.001
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[Accuracy(name='accuracy'),
                                                                                     Precision(name='precision'),
                                                                                     Recall(name='recall'),
                                                                                     AUC(name='auc')])

    def train_model(self) -> History:
        if self.model is None:
            raise RuntimeError("Model has not been built.")
        logger.info(f"Train label mean: {self.label_training_data.mean().round(4)}")
        logger.info(f"Validation label mean: {self.label_validation_data.mean().round(4)}")

        early_stopping: EarlyStopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        )

        self.model.summary()

        history: History = self.model.fit(
            self.batched_training_data,
            self.label_training_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.batched_validation_data, self.label_validation_data),
            callbacks=[early_stopping]
        )

        self.model.save(self.model_save_path)
        logger.info(f"Model saved to:{self.model_save_path}")

        return history


import time
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class StockTrendLSTMModel:
    def __init__(self, input_shape, training_data, training_data_target, validation_data, validation_data_target, epochs: int, batch_size: int):
        self.model = None
        self.input_shape = input_shape
        self.save_name = "StockTrendLSTM"
        self.model_save_path: str = f"../models/saved models/trained_{self.save_name}_model.keras"
        self.training_data = training_data
        self.training_data_target = training_data_target
        self.validation_data = validation_data
        self.validation_data_target = validation_data_target
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=self.input_shape))
        self.model.add(Dropout(0.8))
        self.model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))

        learning_rate = 0.0001
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        start_time = time.time()
        history = self.model.fit(self.training_data, self.training_data_target, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.validation_data, self.validation_data_target), callbacks=[early_stopping])
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        self.model.save(self.model_save_path)
        print(f"Model saved to:", self.model_save_path)

        return history

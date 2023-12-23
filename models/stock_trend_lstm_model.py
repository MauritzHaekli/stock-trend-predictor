import time
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class StockTrendLSTMModel:
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape
        self.save_name = "StockTrendLSTM"
        self.model_save_path: str = f"../models/saved models/trained_{self.save_name}_model.keras"
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=self.input_shape))
        self.model.add(Dropout(0.8))
        self.model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))

        learning_rate = 0.0001
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, training_data, training_data_target, validation_data, validation_data_target, epochs=50, batch_size=8):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        start_time = time.time()
        history = self.model.fit(training_data, training_data_target, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_data_target), callbacks=[early_stopping])
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        self.model.save(self.model_save_path)
        print(f"Model saved to:", self.model_save_path)

        return history

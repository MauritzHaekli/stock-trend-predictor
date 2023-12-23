import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:

    """
    This class is used to preprocess raw stock data in order to optimize neural network performance.

    Attributes:
        data: raw stock data from a csv file
        target column: name of the column for which we want to make a prediction
        validation_size: A number between 0 and 1 to split data into testing and validation sets
        scaler: An sklearn function to preprocess data. Choose for desired purpose
        X_batched: contains (len(data) - lookback_period) entries. The first entry is a list of the first lookback_period rows of data. X_batched[index][-1] gives you the (index + lookback_period)-th row of data
        y_batched: contains (len(df) - seq_length) entries. The first entry is the "target" column of the seq_length row. X[index][-1][-1] and y[index] are the same "target" value.
        If df has 5000 entries, X.shape y.shape will give you: ((4980, 10, 25), (4980,))

        X_train_split: Split unscaled training set derived from X_batched
        y_train_split: Split unscaled training target set derived from y_batched
        X_validation; Split unscaled validation set derived from X_batched
        y_validation_split: Split unscaled training target set derived from y_batched

        X_train_scaled: Scaled training set derived from X_train_split
        X_validation_scaled: Scaled training set derived from X_validation_split
        X_train_scaled: Scaled training set derived from X_batched. Since we want to test the NN on unseen data, no split has been performed prior

    """
    def __init__(self, data: pd.DataFrame, lookback_period: int, target_column: str, validation_size: float):

        self.data = data
        self.lookback_period = lookback_period
        self.target_column = target_column
        self.validation_size = validation_size
        self.scaler = MinMaxScaler()
        self.X_batched = None
        self.y_batched = None
        self.X_train_split = None
        self.y_train_split = None
        self.X_validation_split = None
        self.y_validation_split = None
        self.X_train_scaled = None
        self.X_validation_scaled = None
        self.X_testing_scaled = None

        self.set_lookback_batch()
        self.set_lookback_target()
        self.split_data()
        self.set_scaled_training_lookback_batch()
        self.set_scaled_validation_lookback_batch()
        self.set_scaled_testing_lookback_batch()

    def set_lookback_batch(self):
        time_series_batch = []
        for row in range(len(self.data) - self.lookback_period):
            time_series_batch.append(self.data.iloc[row:row + self.lookback_period].values)

        time_series_batch: [[[]]] = np.array(time_series_batch)
        self.X_batched = time_series_batch

    def set_lookback_target(self):
        time_series_target = []
        for row in range(len(self.data) - self.lookback_period):
            time_series_target.append(self.data[self.target_column].iloc[row + self.lookback_period - 1])

        time_series_target: [] = np.array(time_series_target)
        self.y_batched = time_series_target

    def split_data(self):
        self.X_train_split, self.X_validation_split, self.y_train_split, self.y_validation_split = train_test_split(self.X_batched, self.y_batched, test_size=self.validation_size, random_state=42)

    def set_scaled_training_lookback_batch(self):
        training_data = self.X_train_split
        reshaped = training_data.reshape(-1, training_data.shape[-1])
        scaled = self.scaler.fit_transform(reshaped)
        self.X_train_scaled = scaled.reshape(training_data.shape)

    def set_scaled_validation_lookback_batch(self):
        validation_data = self.X_validation_split
        reshaped = validation_data.reshape(-1, validation_data.shape[-1])
        scaled = self.scaler.fit_transform(reshaped)
        self.X_validation_scaled = scaled.reshape(validation_data.shape)

    def set_scaled_testing_lookback_batch(self):
        testing_data = self.X_batched
        reshaped = testing_data.reshape(-1, testing_data.shape[-1])
        scaled = self.scaler.fit_transform(reshaped)
        self.X_testing_scaled = scaled.reshape(testing_data.shape)

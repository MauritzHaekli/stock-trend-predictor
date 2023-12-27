import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        """
            This class is used to preprocess raw stock data containing price information and technical indicators in order to optimize neural network performance.

            Attributes:
                data: raw stock data from a csv file
                lookback_period: A period with length n we provide for our NN to look back upon, giving it the opportunity to take past periods into account.
                target column: name of the column for which we want to make a prediction.
                validation_size: A number between 0 and 1 to split data into testing and validation sets.
                trend_length: The length that we we try to predict. Trend_length of 10 means we want to predict, what happens in 10 iterations.
                scaler: An sklearn function to preprocess data. Choose for desired purpose.
                target_data: Stock data set by set_target_data(). Datetime column and price columns have been dropped, "target" column has been added and shifted according to the trend length.
                X_batched: contains (len(data) - lookback_period) entries. The first entry is a list of the first lookback_period rows of data. X_batched[index][-1] gives you the (index + lookback_period)-th row of data
                y_batched: contains (len(df) - seq_length) entries. The first entry is the "target" column of the seq_length row.
                If self.data has 5000 entries, X.shape, y.shape will give you: ((4980, 10, 25), (4980,)).

                X_train_split: Split unscaled training set derived from X_batched
                y_train_split: Split unscaled training target set derived from y_batched
                X_validation; Split unscaled validation set derived from X_batched
                y_validation_split: Split unscaled training target set derived from y_batched

                X_train_scaled: Scaled training set derived from X_train_split
                X_validation_scaled: Scaled training set derived from X_validation_split
                X_train_scaled: Scaled training set derived from X_batched. Since we want to test the NN on unseen data, no split has been performed prior

            """

        with open('../config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.data: pd.DataFrame = data
        self.validation_size: float = config["data"]["validation_size"]
        self.lookback_period: int = config["data"]["lookback_period"]
        self.target_column: str = config["data"]["target_column"]
        self.trend_length: int = config["data"]["trend_length"]
        self.trend_columns: [str] = config["data"]["trend_columns"]
        self.scaler = StandardScaler()
        self.trend_data: pd.DataFrame = self.get_trend_data(self.data)
        self.target_data: pd.DataFrame = self.get_target_data(self.trend_data)
        self.X_batched: [[[float]]] = self.get_lookback_batch()
        self.y_batched: [float] = self.get_lookback_target()
        self.X_train_split: [[[float]]] = self.get_split_data()[0]
        self.y_train_split: [float] = self.get_split_data()[2]
        self.X_validation_split: [[[float]]] = self.get_split_data()[1]
        self.y_validation_split: [float] = self.get_split_data()[3]
        self.X_train_scaled: [[[float]]] = self.get_scaled_data()[0]
        self.X_validation_scaled: [[[float]]] = self.get_scaled_data()[1]
        self.X_testing_scaled: [[[float]]] = self.get_scaled_data()[2]

    def get_trend_data(self, time_series_data: pd.DataFrame) -> pd.DataFrame:

        """
        This function takes the raw time series data of a stock and applies calculations about changes, previous data and trends of several columns like "open" and "volume" to each row.
        The "previous {column_name}" columns should contain the entries of the {column_name} a trend_length prior. So with a trend length of 10, ""previous {column_name}" should contain the
        {column_name} entry 10 rows before.
        The "{column_name}-change" columns should contain the difference of the current {column_name} entry minus the {column_name} entry a trend_length ago.
        The "{column_name}-trend" columns should contain a binary indicator (0 or 1) to indicate, if {column_name} has decreased or increased since trend_length ago
        :param time_series_data: A pandas dataframe containing stock data time series 
        :return applied_trend_dataframe: A pandas dataframe in which price comparisons and trends have been added
        """
        trend_dataframe: pd.DataFrame = time_series_data.copy()
        trend_increased: int = 1
        trend_decreased: int = 0
        decimal_places: int = 2

        for column_name in self.trend_columns:
            applied_comparison_column_name: str = f"previous {column_name}"
            applied_change_column_name: str = f"{column_name}-change"
            applied_trend_column_name: str = f"{column_name}-trend"
            not_computable_placeholder: int = 0
            for entry in range(len(time_series_data)):
                if entry < self.trend_length:
                    trend_dataframe.loc[entry, applied_comparison_column_name] = not_computable_placeholder
                    trend_dataframe.loc[entry, applied_change_column_name] = not_computable_placeholder
                    trend_dataframe.loc[entry, applied_trend_column_name] = trend_increased
                elif entry >= self.trend_length:
                    column_previous: float = round(float(trend_dataframe.loc[entry - self.trend_length, f"{column_name}"]), decimal_places)
                    column_current: float = round(float(trend_dataframe.loc[entry, f"{column_name}"]), decimal_places)
                    column_difference: float = round(column_current - column_previous, decimal_places)

                    trend_dataframe.loc[entry, f"previous {column_name}"] = column_previous
                    trend_dataframe.loc[entry, f"{column_name}-change"] = column_difference
                    if trend_dataframe.loc[entry, f"{column_name}"] <= trend_dataframe.loc[entry - self.trend_length, f"{column_name}"]:
                        trend_dataframe.loc[entry, applied_trend_column_name] = trend_decreased
                    else:
                        trend_dataframe.loc[entry, applied_trend_column_name] = trend_increased
            trend_dataframe[applied_trend_column_name] = trend_dataframe[applied_trend_column_name].astype(int)
        return trend_dataframe

    def get_target_data(self, trend_data: pd.DataFrame) -> pd.DataFrame:
        target_data: pd.DataFrame = trend_data.copy()
        prediction_target_column_name: str = 'open-trend'
        columns_to_drop: [str] = ["datetime", "open", "high", "low", "close", "previous open", "open-change", "ema"]
        target_data.index = pd.to_datetime(target_data['datetime'], format='%Y-%m-%d %H:%M:%S')
        target_data['day_of_week'] = target_data.index.day_of_week
        target_data['hour'] = target_data.index.hour
        target_data['target'] = target_data[prediction_target_column_name].shift(-self.trend_length, fill_value=0).astype(int)
        target_data.drop(columns=columns_to_drop, axis=1, inplace=True)
        target_data = target_data.iloc[self.trend_length:]
        return target_data

    def get_lookback_batch(self) -> [[[float]]]:
        time_series_batch = []
        for row in range(len(self.target_data) - self.lookback_period):
            time_series_batch.append(self.target_data.iloc[row:row + self.lookback_period].values)

        time_series_batch: [[[float]]] = np.array(time_series_batch)
        return time_series_batch

    def get_lookback_target(self) -> [float]:
        time_series_target = []
        for row in range(len(self.target_data) - self.lookback_period):
            time_series_target.append(self.target_data[self.target_column].iloc[row + self.lookback_period - 1])

        time_series_target: [float] = np.array(time_series_target)
        return time_series_target

    def get_split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train_split, x_validation_split, y_train_split, y_validation_split = train_test_split(self.X_batched, self.y_batched, test_size=self.validation_size, random_state=42)
        return x_train_split, x_validation_split, y_train_split, y_validation_split

    def get_scaled_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Fit the scaler only on the training data
        self.scaler.fit(self.X_train_split.reshape(-1, self.X_train_split.shape[-1]))

        # Transform the training, validation, and testing data using the same scaler
        x_train_scaled = self.scaler.transform(self.X_train_split.reshape(-1, self.X_train_split.shape[-1])).reshape(self.X_train_split.shape)
        x_validation_scaled = self.scaler.transform(self.X_validation_split.reshape(-1, self.X_validation_split.shape[-1])).reshape(self.X_validation_split.shape)
        x_testing_scaled = self.scaler.transform(self.X_batched.reshape(-1, self.X_batched.shape[-1])).reshape(self.X_batched.shape)

        return x_train_scaled, x_validation_scaled, x_testing_scaled


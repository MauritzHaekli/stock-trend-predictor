import pandas as pd
from backend.utils.feature_column_names import FeatureColumnNames


class BaseValidator:
    """Base class providing common validation methods."""

    @staticmethod
    def validate_positive_integer(value, name):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def validate_series(feature: pd.Series):
        if not isinstance(feature, pd.Series):
            raise ValueError("The feature must be a pandas Series.")
        if feature.empty:
            raise ValueError("The feature cannot be empty.")
        if feature.name is None:
            raise ValueError("The feature must have a name attribute.")


class FeatureValidator(BaseValidator):
    """Validator for feature-specific logic."""

    def validate_feature_provider_input(self, time_series: pd.DataFrame, periods: int, rounding_factor: int, cutoff: int):
        self.validate_dataframe(time_series)
        self.validate_positive_integer(periods, "periods")
        self.validate_positive_integer(rounding_factor, "rounding_factor")
        self.validate_positive_integer(cutoff, "cutoff")
        self.validate_required_columns(time_series)

    @staticmethod
    def validate_dataframe(dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("time_series must be a pandas DataFrame.")

    @staticmethod
    def validate_required_columns(time_series: pd.DataFrame):
        feature_column_names: FeatureColumnNames = FeatureColumnNames()

        required_columns = [
            feature_column_names.DATETIME, feature_column_names.OPEN_PRICE,
            feature_column_names.HIGH_PRICE, feature_column_names.LOW_PRICE,
            feature_column_names.CLOSE_PRICE, feature_column_names.VOLUME
        ]
        missing_columns = [col for col in required_columns if col not in time_series.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


class PeriodValidator(BaseValidator):
    """Validator for period-specific logic."""
    @staticmethod
    def validate_periods(periods: int, feature: pd.Series):
        if not isinstance(periods, int):
            raise ValueError("Periods must be an integer.")
        if periods <= 0 or periods >= len(feature):
            raise ValueError("Periods must be a positive integer smaller than the length of the feature")


class DatetimeValidator(BaseValidator):

    @staticmethod
    def validate_datetime_feature(time_series):
        feature_column_names: FeatureColumnNames = FeatureColumnNames()

        if not isinstance(time_series, pd.DataFrame):
            raise ValueError("time_series must be a pandas DataFrame containing a datetime column.")
        if feature_column_names.DATETIME not in time_series.columns:
            raise ValueError(f"Expected column '{feature_column_names.DATETIME}' not found in time_series.")
        if time_series[feature_column_names.DATETIME].isna().any():
            raise ValueError(f"Column '{feature_column_names.DATETIME}' contains missing values.")

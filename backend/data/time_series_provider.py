import requests
import yaml
import pandas as pd


def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """
    Saves a given DataFrame to a CSV file at the specified path.

    :param dataframe: DataFrame to be saved to CSV.
    :param output_path: Path where the CSV file will be saved.
    """
    dataframe.to_csv(output_path, index=False)
    print(f"File saved to: {output_path}")


class TimeSeriesProvider:
    """
    Fetches time series stock data from the Twelve Data API, processes it into a DataFrame,
    and saves it as a CSV file.

    The TimeSeriesProvider uses configuration parameters from the config.yaml file to define the API
    endpoint, request parameters, and file output path. It provides methods to fetch data
    (`get_time_series`) and to automate the fetching and saving process (`generate_csv`).

    Attributes:
        api_key (str): API key for authentication with the Twelve Data API.
        time_series_key (str): Specifies the time series indicator to retrieve, in this case stock data.
        interval (str): Time interval for the time series data (e.g., '1min', '1h').
        order (str): Order of time series data (ASC & DESC).
        symbol (str): Stock symbol for which to retrieve data.
        output_size (int): The maximum number of data points to retrieve (Max. 5000).
        decimal_places (int): Number of decimal places for data precision.
        dataframe_key (str): Key used for datetime values in the DataFrame.
        response_values_key (str): Key to access the data values in the API response.
    """
    def __init__(self):
        with open('../../config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
            time_series_provider_parameters = config['time_series_provider_parameters']

        self.api_key = time_series_provider_parameters.get('api_key')
        self.time_series_key = time_series_provider_parameters.get('time_series_key')
        self.interval = time_series_provider_parameters.get('interval')
        self.order = time_series_provider_parameters.get('order')
        self.symbol = time_series_provider_parameters.get('symbol')
        self.output_size = time_series_provider_parameters.get('output')
        self.decimal_places = time_series_provider_parameters.get('decimal_places')
        self.dataframe_key = "datetime"
        self.response_values_key = "values"

    def get_time_series(self) -> pd.DataFrame:

        """
        Fetches time series data from the Twelve Data API and returns it as a pandas DataFrame.

        Constructs the request URL based on configuration settings, sends the request,
        and processes the response JSON into a DataFrame.

        :return: A DataFrame with the time series data, or an empty DataFrame if the request fails.
        """

        url = (f"https://api.twelvedata.com/{self.time_series_key}?"
               f"symbol={self.symbol}"
               f"&interval={self.interval}"
               f"&order={self.order}"
               f"&dp={self.decimal_places}"
               f"&outputsize={self.output_size}"
               f"&apikey={self.api_key}"
               )
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return pd.DataFrame()

        values = response.json().get(self.response_values_key, [])
        dataframe = pd.DataFrame(values)
        dataframe[self.dataframe_key] = pd.to_datetime(dataframe[self.dataframe_key])
        return dataframe

    def generate_csv(self):

        """
        Fetches time series data and saves it as a CSV file.

        Calls `get_time_series` to retrieve the data as a DataFrame and, if the DataFrame
        is not empty, saves it to a CSV file using `save_to_csv`. The output file path
        is constructed based on the configuration settings.
        """
        time_series_df = self.get_time_series()
        if not time_series_df.empty:
            save_file_path = f"../main/data/twelvedata/time series ({self.interval})/{self.symbol}_time_series.csv"
            save_to_csv(time_series_df, save_file_path)


if __name__ == "__main__":
    generator = TimeSeriesProvider()
    generator.generate_csv()

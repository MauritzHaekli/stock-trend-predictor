import yaml
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volume import VolumePriceTrendIndicator


with open('../config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

    stock_symbols = config['preprocess']['stock_symbols']


def generate_feature_dataframe(symbols: [str]):

    for symbol in symbols:

        interval: str = "1h"
        csv_file_path: str = f"../data/twelvedata/time series ({interval})/{symbol}_time_series.csv"

        feature_data: pd.DataFrame = pd.read_csv(csv_file_path)

        bollinger_period: int = 20
        bollinger_std: int = 2
        atr_window: int = 14
        macd_short_period: int = 12
        macd_long_period: int = 26
        macd_signal_period: int = 9
        adx_period: int = 14
        sma_period: int = 9
        ema_period: int = 9
        rsi_period: int = 14
        stoch_window: int = 14
        stoch_smooth_window: int = 3

        bb = BollingerBands(close=feature_data['close'], window=bollinger_period, window_dev=bollinger_std, fillna=True)
        atr = AverageTrueRange(high=feature_data['high'], low=feature_data['low'], close=feature_data['close'], window=atr_window, fillna=True)
        macd = MACD(close=feature_data['close'], window_fast=macd_short_period, window_slow=macd_long_period, window_sign=macd_signal_period, fillna=True)
        adx = ADXIndicator(high=feature_data['high'], low=feature_data['low'], close=feature_data['close'], window=adx_period, fillna=True)
        sma = SMAIndicator(close=feature_data['close'], window=sma_period, fillna=True)
        ema = EMAIndicator(close=feature_data['close'], window=ema_period, fillna=True)
        rsi = RSIIndicator(feature_data['close'], window=rsi_period, fillna=True)
        stoch = StochasticOscillator(high=feature_data['high'], low=feature_data['low'], close=feature_data['close'], window=stoch_window, smooth_window=stoch_smooth_window, fillna=True)
        vpt = VolumePriceTrendIndicator(close=feature_data['close'], volume=feature_data['volume'], fillna=True)

        feature_data['percent_b'] = (feature_data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()).round(4)
        feature_data['atr'] = atr.average_true_range().round(4)
        feature_data['macd'] = macd.macd().round(4)
        feature_data['macd_signal'] = macd.macd_signal().round(4)
        feature_data['macd_hist'] = macd.macd_diff().round(4)
        feature_data['adx'] = adx.adx().round(4)
        feature_data['sma'] = sma.sma_indicator().round(4)
        feature_data['ema'] = ema.ema_indicator().round(4)
        feature_data['rsi'] = rsi.rsi().round(4)
        feature_data['%K'] = stoch.stoch().round(4)
        feature_data['%D'] = stoch.stoch_signal().round(4)
        feature_data['vpt'] = vpt.volume_price_trend().round(4)

        feature_data.index = pd.to_datetime(feature_data['datetime'], format='%Y-%m-%d %H:%M:%S')
        feature_data['day'] = feature_data.index.day_of_week
        feature_data['hour'] = feature_data.index.hour

        feature_data = feature_data.copy()

        feature_data['open change'] = feature_data['open'].pct_change().round(4)
        feature_data['high change'] = feature_data['high'].pct_change().round(4)
        feature_data['low change'] = feature_data['low'].pct_change().round(4)
        feature_data['close change'] = feature_data['close'].pct_change().round(4)

        feature_data['price movement'] = (feature_data['close'] - feature_data['open']).round(4)
        feature_data['price range'] = (feature_data['high'] - feature_data['low']).round(4)
        feature_data['price trend'] = (feature_data['price movement'] > 0).astype(int)

        feature_data['open trend'] = (feature_data['open'] > feature_data['open'].shift(1)).astype(int)
        feature_data['high trend'] = (feature_data['high'] > feature_data['high'].shift(1)).astype(int)
        feature_data['low trend'] = (feature_data['low'] > feature_data['low'].shift(1)).astype(int)
        feature_data['close trend'] = (feature_data['close'] > feature_data['close'].shift(1)).astype(int)
        feature_data['volume trend'] = (feature_data['volume'] > feature_data['volume'].shift(1)).astype(int)

        feature_data = feature_data[30:]

        save_file_path: str = f"../data/twelvedata/feature time series ({interval})/{symbol}_feature_time_series.csv"
        feature_data.to_csv(save_file_path, index=False)
        print(f"File saved to: {save_file_path}")


if __name__ == "__main__":
    generate_feature_dataframe(stock_symbols)
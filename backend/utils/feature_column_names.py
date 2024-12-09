from enum import Enum


class FeatureColumnNames:
    DATETIME = "datetime"

    OPEN_PRICE = "open"
    HIGH_PRICE = "high"
    LOW_PRICE = "low"
    CLOSE_PRICE = "close"

    SMA = "sma"
    SMA_SLOPE = "sma slope"

    EMA = "ema"
    EMA_SLOPE = "ema slope"

    VOLUME = "volume"

    PERCENT_B = "percent_b"
    ATR = "atr"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"
    MACD_HIST = "macd_hist"
    ADX = "adx"
    RSI = "rsi"

    DAY = "day"
    HOUR = "hour"

    PRICE_MOVEMENT = "price movement"
    PRICE_RANGE = "price range"
    PRICE_TREND = "price trend"

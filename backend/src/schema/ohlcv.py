class OHLCVColumns:
    DATETIME = "datetime"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"

    OHLCV = (OPEN, HIGH, LOW, CLOSE, VOLUME)
    OHLC = (OPEN, HIGH, LOW, CLOSE)
    ALL = (DATETIME, *OHLCV)
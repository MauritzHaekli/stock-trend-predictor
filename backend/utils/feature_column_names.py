class FeatureColumnNames:
    def __init__(self):
        self.datetime: str = "datetime"

        self.open_price: str = "open"
        self.high_price: str = "high"
        self.low_price: str = "low"
        self.close_price: str = "close"

        self.sma_price: str = "sma"
        self.sma_slope: str = "sma slope"

        self.ema_price: str = "ema"
        self.ema_slope: str = "ema slope"

        self.volume: str = "volume"

        self.percent_b: str = "percent_b"
        self.atr: str = "atr"
        self.macd: str = "macd"
        self.macd_signal: str = "macd_signal"
        self.macd_hist: str = "macd_hist"
        self.adx: str = "adx"
        self.rsi: str = "rsi"

        self.day: str = "day"
        self.hour: str = "hour"

        self.price_movement: str = "price movement"
        self.price_range: str = "price range"
        self.price_trend: str = "price trend"

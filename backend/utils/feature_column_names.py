class FeatureColumnNames:
    def __init__(self):
        self.datetime: str = "datetime"

        self.open_price: str = "open"
        self.high_price: str = "high"
        self.low_price: str = "low"
        self.close_price: str = "close"

        self.sma_price: str = "sma"
        self.sma_absolute_change: str = "sma change"
        self.sma_percent_change: str = "sma change (%)"
        self.sma_trend: str = "sma trend"

        self.ema_price: str = "ema"
        self.ema_absolute_change: str = "ema change"
        self.ema_percent_change: str = "ema change (%)"
        self.ema_trend: str = "ema trend"

        self.volume: str = "volume"

        self.percent_b: str = "percent_b"
        self.atr: str = "atr"
        self.macd: str = "macd"
        self.macd_signal: str = "macd_signal"
        self.macd_hist: str = "macd_hist"
        self.adx: str = "adx"
        self.rsi: str = "rsi"
        self.fast_stochastic: str = "%K"
        self.slow_stochastic: str = "%D"
        self.vpt: str = "vpt"

        self.day: str = "day"
        self.hour: str = "hour"

        self.price_movement: str = "price movement"
        self.price_range: str = "price range"
        self.price_trend: str = "price trend"

import pandas_ta as ta

TrendStrategy = ta.Strategy(
    name="Trend Signal Strategy",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        {"kind": "sma", "length": 50},
        {"kind": "ema", "length": 10},
        {"kind": "ema", "length": 20},
        {"kind": "donchian", "lower_length": 10, "upper_length": 10},
        {"kind": "donchian", "lower_length": 50, "upper_length": 50},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
        # additional indicators
        {"kind": "adx", "length": 20, "scalar": 1},
        {"kind": "aroon", "length": 20, "scalar": 1, "talib": False},
        {"kind": "stc", "tclength": 10, "fast": 10, "slow": 20},
        {"kind": "natr", "length": 20, "scalar": 1, "talib": False},
        {"kind": "rsi", "length": 20, "scalar": 1, "talib": False},
        {"kind": "cci", "length": 20},
    ],
)

CommonStrategy = ta.Strategy(
    name="Standard Technical Indicator in various research of automatic stock trading",
    description="SMA MACD RSI CCI ADX Bollinger",
    ta=[
        {"kind": "sma", "length": 10},
        {"kind": "rsi", "length": 14},
        {"kind": "cci", "length": 14},
        {"kind": "adx", "length": 14},
        {"kind": "bbands"},
        {"kind": "macd"},
    ],
)

SimpleStrategy = ta.Strategy(
    name="Simple Strategy for testing",
    description="SMA RSI",
    ta=[
        {"kind": "sma", "length": 10},
        {"kind": "rsi", "length": 14},
    ],
)

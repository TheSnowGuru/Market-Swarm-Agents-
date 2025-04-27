# ┌──────────┐
# │ config.py │
# └──────────┘
SWING_LENGTH = 15            # bars for swing detection
CONFIRMATION_BARS = 2        # buffer bars between sweep & FVG
LIQ_WINDOW_BARS = SWING_LENGTH + CONFIRMATION_BARS + 2  # lookback for liquidity sweep

ATR_LENGTH = 14              # ATR period
ATR_SMA_LENGTH = 20          # ATR SMA period for volatility filter

HTF_EMA_SLOW = 20            # EMA period on daily for trend filter

# trading hours filter (HH:MM format)
TIME_START = "09:30"
TIME_END = "16:00"


# ┌────────────┐
# │ filters.py │
# └────────────┘
import talib
from config import ATR_LENGTH, ATR_SMA_LENGTH, HTF_EMA_SLOW, LIQ_WINDOW_BARS, TIME_START, TIME_END

def htf_trend_ok(df):
    # Daily EMA slope
    htf = df.resample("1D").agg({"close": "last"}).dropna()
    ema = talib.EMA(htf["close"], timeperiod=HTF_EMA_SLOW)
    return ema.iloc[-1] > ema.iloc[-2]


def atr_filter(df):
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=ATR_LENGTH)
    sma = atr.rolling(ATR_SMA_LENGTH).mean()
    return atr.iloc[-1] > sma.iloc[-1]


def fvg_fresh(df, bottom, top):
    recent = df.iloc[-LIQ_WINDOW_BARS:]
    # ensure no prior candle already touched the gap
    return not ((recent["low"] >= bottom) & (recent["low"] <= top)).any()


def time_filter(bar):
    t = bar.name.time().strftime("%H:%M")
    return TIME_START <= t <= TIME_END


# ┌──────────────┐
# │ strategy.py │
# └──────────────┘
import smc
from config import SWING_LENGTH, LIQ_WINDOW_BARS
from filters import htf_trend_ok, atr_filter, fvg_fresh, time_filter

# storage for pending FVG setups per symbol
fvg_storage = {}

def generate_signals(df, symbol):
    swings = smc.swing_highs_lows(df, swing_length=SWING_LENGTH)
    fvg = smc.fvg(df, join_consecutive=True)
    liq = smc.liquidity(df, swings)
    signals = []
    
    # check setup at bar -2
    idx = -2
    # LONG setup: bullish FVG + prior bearish sweep
    if fvg.iloc[idx] == 1 and liq.iloc[-LIQ_WINDOW_BARS:idx].eq(-1).any():
        zone = fvg.zone.iloc[idx]  # (bottom, top)
        fvg_storage[symbol] = {"zone": zone, "bar": df.index[idx]}
    # SHORT setup: bearish FVG + prior bullish sweep
    if fvg.iloc[idx] == -1 and liq.iloc[-LIQ_WINDOW_BARS:idx].eq(1).any():
        zone = fvg.zone.iloc[idx]
        fvg_storage[symbol] = {"zone": zone, "bar": df.index[idx]}
    
    # Entry check on the latest closed bar
    last = df.iloc[-1]
    setup = fvg_storage.get(symbol)
    if setup:
        bottom, top = setup["zone"]
        # LONG entry
        if bottom < last["low"] <= top and last["close"] > last["open"]:
            if htf_trend_ok(df) and atr_filter(df) and fvg_fresh(df, bottom, top) and time_filter(last):
                signals.append(("LONG", last.name, bottom, top))
                fvg_storage.pop(symbol)
        # SHORT entry
        if bottom <= last["high"] < top and last["close"] < last["open"]:
            if not htf_trend_ok(df) and atr_filter(df) and fvg_fresh(df, bottom, top) and time_filter(last):
                signals.append(("SHORT", last.name, bottom, top))
                fvg_storage.pop(symbol)
    
    return signals


# ┌──────────┐
# │ main.py │
# └──────────┘
import pandas as pd
time
from broker_api import Broker  # your broker wrapper
from strategy import generate_signals

# initialize empty DataFrames for each symbol
symbols = ["EURUSD", "GBPUSD"]  # replace with your symbols
symbol_dfs = {sym: pd.DataFrame(columns=["open","high","low","close","volume"]) for sym in symbols}

broker = Broker()

def on_new_candle(event):
    sym = event["symbol"]
    candle = event["candle"]  # dict with keys as df columns
    df = symbol_dfs[sym].append(candle, ignore_index=False)
    symbol_dfs[sym] = df
    signals = generate_signals(df, sym)
    for side, ts, bot, top in signals:
        broker.place_order(symbol=sym, side=side, price=None)
        print(f"{side} signal on {sym} at {ts}, FVG zone {bot}-{top}")


def run():
    # stream 15m candles from broker
    for evt in broker.stream_candles(timeframe="15m"):
        if evt.get("is_closed"):
            on_new_candle(evt)

if __name__ == "__main__":
    run()

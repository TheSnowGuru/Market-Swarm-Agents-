def candle_abbrev(ohlc, decimals=2):
    """
    Create a single abbreviated string that tells the candle story with precise decimal percentages.

    Parameters:
        ohlc (dict): Dictionary with keys 'open', 'high', 'low', 'close'.
        decimals (int): Number of decimal places for percentages.

    Returns:
        str: Abbreviated candle representation.
    """
    open_ = ohlc['open']
    high = ohlc['high']
    low = ohlc['low']
    close = ohlc['close']

    total_range = high - low
    if total_range == 0:
        raise ValueError("High and Low are identical; can't calculate percentages.")

    def fmt(value):
        # Format percentage with specified decimal places, strip trailing zeros
        s = f"{value:.{decimals}f}".rstrip('0').rstrip('.')
        return s

    # Doji candle: no body, mark as 'd'
    if open_ == close:
        head = high - close
        tail = open_ - low
        head_perc = (head / total_range) * 100
        tail_perc = (tail / total_range) * 100
        return f"t{fmt(tail_perc)}-d-h{fmt(head_perc)}-d"
    else:
        if close > open_:
            candle_color = "g"
            head = high - close
            body = close - open_
            tail = open_ - low
        else:
            candle_color = "r"
            head = high - open_
            body = open_ - close
            tail = close - low

        head_perc = (head / total_range) * 100
        body_perc = (body / total_range) * 100
        tail_perc = (tail / total_range) * 100

        return f"t{fmt(tail_perc)}-b{fmt(body_perc)}-h{fmt(head_perc)}-{candle_color}"

if __name__ == "__main__":
    # Example for a small head percentage
    ohlc_small_head = {
        'high': 1.301,
        'open': 1.205,
        'close': 1.300,
        'low': 1.202
    }
    print(candle_abbrev(ohlc_small_head, decimals=3))

    # Example doji
    ohlc_doji = {
        'high': 1.305,
        'open': 1.250,
        'close': 1.250,
        'low': 1.245
    }
    print(candle_abbrev(ohlc_doji, decimals=3))

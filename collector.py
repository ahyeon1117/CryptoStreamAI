import requests
import pandas as pd
import time

def get_binance_ohlcv(symbol="BTCUSDT", interval="1m", days=7):
    all_data = []
    base_url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    interval_ms = 60 * 1000  # 1m
    max_limit = 1000
    total_candles = days * 24 * 60
    rounds = total_candles // max_limit + 1

    for _ in range(rounds):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": max_limit,
            "endTime": end_time
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            break

        all_data = data + all_data
        end_time = data[0][0] - 1
        time.sleep(0.2)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]


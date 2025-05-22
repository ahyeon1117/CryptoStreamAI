import requests
import pandas as pd

def fetch_5min_ohlcv(symbol, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "5m", "limit": limit}
    response = requests.get(url, params=params).json()
    df = pd.DataFrame(response, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'NumTrades',
        'TakerBuyVol', 'TakerBuyQuoteVol', 'Ignore'
    ])
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteAssetVolume']] = \
        df[['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteAssetVolume']].astype(float)
    return df[['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'QuoteAssetVolume']]
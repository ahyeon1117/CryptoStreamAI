import requests

def get_top_100_futures_coins():
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    response = requests.get(url).json()
    usdt_pairs = [coin for coin in response if coin['symbol'].endswith('USDT')]
    sorted_coins = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [(coin['symbol'], float(coin['quoteVolume'])) for coin in sorted_coins[:100]]


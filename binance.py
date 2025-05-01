from binance.client import Client
from config import API_KEY, API_SECRET

client = Client(API_KEY, API_SECRET)

def get_balance(asset="USDT"):
    balance = client.get_asset_balance(asset=asset)
    return float(balance['free'])

def market_buy(symbol, usdt_amount):
    price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    quantity = round(usdt_amount / price, 5)
    return client.order_market_buy(symbol=symbol, quantity=quantity)

def market_sell(symbol, btc_amount):
    return client.order_market_sell(symbol=symbol, quantity=btc_amount)
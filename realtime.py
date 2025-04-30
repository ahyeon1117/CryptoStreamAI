import asyncio
import websockets
import json
import numpy as np
from lstm_model import scaler

seq = []  # 최근 60개 가격

async def realtime_predict(model):
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(uri) as websocket:
        while True:
            msg = await websocket.recv()
            price = float(json.loads(msg)['p'])

            seq.append(price)
            if len(seq) > 60:
                seq.pop(0)
                scaled_seq = scaler.transform(np.array(seq).reshape(-1, 1))
                input_seq = np.expand_dims(scaled_seq, axis=0)
                pred = model.predict(input_seq, verbose=0)[0][0]
                pred = scaler.inverse_transform([[pred]])[0][0]
                print(f"📈 실시간 가격: {price:.2f} | 🤖 예측: {pred:.2f}")

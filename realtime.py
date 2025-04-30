import asyncio
import websockets
import json
import numpy as np
from datetime import datetime

def get_empty_seq():
    return []

async def realtime_predict(model, scaler):
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    seq = get_empty_seq()

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                print("✅ WebSocket 연결됨")
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
                        expected_pct = (pred - price) / price

                        if expected_pct >= 0.003:
                            decision = "🔺 Long 포지션 추천"
                        elif expected_pct <= -0.003:
                            decision = "🔻 Short 포지션 추천"
                        else:
                            decision = "⏸️ Hold (관망)"

                        now = datetime.now().strftime('%H:%M:%S')
                        print(f"🕒 {now} | 📈 현재가: {price:.2f} | 🤖 예측: {pred:.2f} | 📊 예측 수익률: {expected_pct*100:.3f}% | 🧠 전략: {decision}")

        except Exception as e:
            print(f"🔌 WebSocket 오류: {e}, 5초 후 재연결...")
            await asyncio.sleep(5)

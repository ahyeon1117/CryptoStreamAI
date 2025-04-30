import asyncio
import websockets
import json
import numpy as np
import csv
from datetime import datetime
from lstm_model import build_lstm_model, create_sequences, scaler
from collector import get_binance_ohlcv
import tensorflow as tf

import platform

# 1. Windows 안정성 확보 (이벤트 루프 정책 설정)
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ GPU 사용 확인
print("✅ GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

# ✅ 모델 학습
df = get_binance_ohlcv(days=7)
prices = df['close'].values.reshape(-1, 1)
scaled = scaler.fit_transform(prices)
X, y = create_sequences(scaled)
model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=20, batch_size=248)

# 📈 실시간 예측 시퀀스 초기화
seq = []
leverage = 100
stop_loss_pct = 0.01
position_size = 100
in_position = False
entry_price = 0

# 거래 로그 초기화
log_file = open("trade_log.csv", mode="a", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["timestamp", "type", "price", "pnl_pct"])

async def realtime_bot():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(uri) as ws:
        print("🚀 실시간 LSTM 매매 봇 (롱 포지션 전용, 모의 테스트) 실행 중...")
        global in_position, entry_price

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                price = float(data['p'])
                ts = datetime.fromtimestamp(data['T'] / 1000.0, tz=datetime.timezone.utc).isoformat()
                seq.append(price)

                if len(seq) < 60:
                    continue
                if len(seq) > 60:
                    seq.pop(0)

                scaled_seq = scaler.transform(np.array(seq).reshape(-1, 1))
                input_seq = np.expand_dims(scaled_seq, axis=0)
                pred = model.predict(input_seq, verbose=0)[0][0]
                pred = scaler.inverse_transform([[pred]])[0][0]

                expected_pct = (pred - price) / price

                if not in_position and expected_pct >= 0.0025:
                    in_position = True
                    entry_price = price
                    print(f"✅ LONG 진입 | 진입가: {entry_price:.2f} | 예측가: {pred:.2f} | 기대 수익률: {expected_pct*100:.2f}%")
                    log_writer.writerow([ts, "BUY", entry_price, ""])
                    log_file.flush()

                elif in_position:
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct <= -stop_loss_pct:
                        print(f"❌ 손절 | 현재가: {price:.2f} | 손실률: {pnl_pct*100:.2f}%")
                        in_position = False
                        log_writer.writerow([ts, "SELL", price, round(pnl_pct * 100, 4)])
                        log_file.flush()
                    elif pnl_pct >= 0.005:
                        print(f"🎉 익절 | 현재가: {price:.2f} | 수익률: {pnl_pct*100:.2f}%")
                        in_position = False
                        log_writer.writerow([ts, "SELL", price, round(pnl_pct * 100, 4)])
                        log_file.flush()
                    else:
                        print(f"📊 보유 중 | 현재가: {price:.2f} | 수익률: {pnl_pct*100:.2f}%")
                else:
                    print(f"⏸️ 관망 | 현재가: {price:.2f} | 예측가: {pred:.2f} | 기대 수익률: {expected_pct*100:.2f}%")

            except Exception as e:
                print("❌ 오류 발생:", e)
                await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(realtime_bot())
    finally:
        log_file.close()

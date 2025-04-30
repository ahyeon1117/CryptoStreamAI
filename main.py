import asyncio
import websockets
import json
import numpy as np
import csv
from datetime import datetime, timezone
from lstm_model import build_lstm_model, create_sequences, scaler
from collector import get_binance_ohlcv
import tensorflow as tf

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
position_type = None  # 'long' or 'short'

# 거래 로그 초기화
log_file = open("trade_log.csv", mode="a", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["timestamp", "type", "price", "pnl_pct"])

async def realtime_bot():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(uri) as ws:
        print("🚀 실시간 LSTM 매매 봇 (롱/숏 포지션 모의 테스트) 실행 중...")
        global in_position, entry_price, position_type

        while True:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                price = float(data['p'])
                ts = datetime.fromtimestamp(data['T'] / 1000.0, tz=timezone.utc).isoformat()
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

                if not in_position:
                    if expected_pct >= 0.0025:
                        in_position = True
                        entry_price = price
                        position_type = 'long'
                        print(f"✅ LONG 진입 | 진입가: {entry_price:.2f} | 예측가: {pred:.2f} | 기대 수익률: {expected_pct*100:.2f}%")
                        log_writer.writerow([ts, "BUY_LONG", entry_price, ""])
                        log_file.flush()
                    elif expected_pct <= -0.0025:
                        in_position = True
                        entry_price = price
                        position_type = 'short'
                        print(f"✅ SHORT 진입 | 진입가: {entry_price:.2f} | 예측가: {pred:.2f} | 기대 수익률: {expected_pct*100:.2f}%")
                        log_writer.writerow([ts, "SELL_SHORT", entry_price, ""])
                        log_file.flush()

                elif in_position:
                    if position_type == 'long':
                        pnl_pct = (price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - price) / entry_price

                    if pnl_pct <= -stop_loss_pct:
                        print(f"❌ 손절 | 현재가: {price:.2f} | 손실률: {pnl_pct*100:.2f}% [{position_type.upper()} 종료]")
                        log_writer.writerow([ts, f"EXIT_{position_type.upper()}", price, round(pnl_pct * 100, 4)])
                        log_file.flush()
                        in_position = False
                    elif pnl_pct >= 0.005:
                        print(f"🎉 익절 | 현재가: {price:.2f} | 수익률: {pnl_pct*100:.2f}% [{position_type.upper()} 종료]")
                        log_writer.writerow([ts, f"EXIT_{position_type.upper()}", price, round(pnl_pct * 100, 4)])
                        log_file.flush()
                        in_position = False
                    else:
                        print(f"📊 보유 중 | 현재가: {price:.2f} | 수익률: {pnl_pct*100:.2f}% [{position_type.upper()}]")

            except Exception as e:
                print("❌ 오류 발생:", e)
                await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(realtime_bot())
    finally:
        log_file.close()
        # 롱/숏 수익 분석
        import pandas as pd
        df = pd.read_csv("trade_log.csv")
        df = df.dropna()
        df_long = df[df['type'].str.contains("LONG") & df['type'].str.startswith("EXIT")]
        df_short = df[df['type'].str.contains("SHORT") & df['type'].str.startswith("EXIT")]
        long_profit = df_long['pnl_pct'].sum()
        short_profit = df_short['pnl_pct'].sum()
        print(f"📊 전략별 총 수익률:")
        print(f"🔺 롱 수익률 합계: {long_profit:.2f}%")
        print(f"🔻 숏 수익률 합계: {short_profit:.2f}%")


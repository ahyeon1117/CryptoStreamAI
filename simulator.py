from collector import get_binance_ohlcv
from lstm_model import build_lstm_model, create_sequences, scaler
import numpy as np
import csv
from tqdm import tqdm

# 1. Load data and train model
df = get_binance_ohlcv(days=7)
prices = df['close'].values.reshape(-1, 1)
scaled = scaler.fit_transform(prices)
X, y = create_sequences(scaled)
model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=5, batch_size=64)  # 더 많은 학습으로 정확도 향상

# 2. Run simulation
results = []
initial_balance = 1000.0
balance = initial_balance
position_size = 100  # USD
leverage = 50  # ⚖️ 50x leverage for more realistic risk
stop_loss_pct = 0.015  # 손절 기준도 조금 더 유연하게 조정  # 2% 손절 기준

for i in tqdm(range(60, len(prices) - 3), desc="Simulating", unit="step"):
    seq = prices[i-60:i]
    current = float(prices[i][0])
    future = float(prices[i+3][0])
    scaled_seq = scaler.transform(seq)
    input_seq = np.expand_dims(scaled_seq, axis=0)
    pred = model.predict(input_seq, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred]])[0][0]

    expected_pct = (pred - current) / current
    if expected_pct >= 0.004:  # 더 보수적인 기준으로 확률 향상
        direction = "long"
        raw_pct = (future - current) / current
        pnl_pct = max(raw_pct, -stop_loss_pct)
    elif expected_pct <= -0.004:
        direction = "short"
        raw_pct = (current - future) / current
        pnl_pct = max(raw_pct, -stop_loss_pct)
    else:
        continue

    is_liquidated = pnl_pct == -stop_loss_pct
    pnl = pnl_pct * position_size * leverage
    balance += pnl
    results.append([
        df['timestamp'].iloc[i],
        current,
        future,
        pnl_pct * 100 * leverage,
        direction,
        pnl > 0,
        round(balance, 2),
        is_liquidated
    ])

# 3. Save results
import pandas as pd

with open("sim_result.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "entry", "exit", "pnl(%)", "direction", "hit", "balance", "liquidated"])
    writer.writerows(results)

# 4. Analyze liquidations and losses
result_df = pd.read_csv("sim_result.csv")
num_liquidated = result_df["liquidated"].sum()
print(f"❌ 청산된 횟수: {int(num_liquidated)}회")

# 손실 집계
loss_trades = result_df[result_df["pnl(%)"] < 0]
total_loss = loss_trades["pnl(%)"].sum()
avg_loss = loss_trades["pnl(%)"].mean()
print(f"💸 총 손실률 합계: {total_loss:.2f}%")
print(f"📉 평균 손실률: {avg_loss:.2f}%")

# 승/패 집계
num_wins = result_df["hit"].sum()
num_losses = len(result_df) - num_wins
print(f"✅ 수익 낸 거래 수: {int(num_wins)}회")
print(f"❌ 손실 낸 거래 수: {int(num_losses)}회")

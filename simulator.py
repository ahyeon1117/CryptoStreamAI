import tensorflow as tf
print("✅ GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

from collector import get_binance_ohlcv
from lstm_model import build_lstm_model, create_sequences, scaler
import numpy as np
import csv
from tqdm import tqdm

# 시각화용 폰트 설정 (한글 깨짐 방지)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. Load data and train model
df = get_binance_ohlcv(days=7)
prices = df['close'].values.reshape(-1, 1)
scaled = scaler.fit_transform(prices)
X, y = create_sequences(scaled)
model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=20, batch_size=248)  # GPU 성능 활용을 위한 배치 최적화  # 더 많은 학습으로 정확도 향상

# 2. Run simulation (optimized with batch prediction)
results = []
initial_balance = 1000.0
balance = initial_balance
position_size = 100  # USD
leverage = 100  # ⚖️ 50x leverage for more realistic risk
stop_loss_pct = 0.01  # 손절 기준을 강화하여 리스크를 더 줄임  # 손절 기준도 조금 더 유연하게 조정  # 2% 손절 기준

# Prepare sequences for batch prediction
sim_inputs = []
currents = []
futures = []
timestamps = []

for i in range(60, len(prices) - 3):
    seq = prices[i-60:i]
    current = float(prices[i][0])
    future = float(prices[i+3][0])
    scaled_seq = scaler.transform(seq)
    sim_inputs.append(scaled_seq)
    currents.append(current)
    futures.append(future)
    timestamps.append(df['timestamp'].iloc[i])

sim_inputs = np.array(sim_inputs)
preds = model.predict(sim_inputs, verbose=0)
preds = scaler.inverse_transform(preds)

for i in tqdm(range(len(sim_inputs)), desc="Simulating", unit="step"):
    pred = preds[i][0]
    current = currents[i]
    future = futures[i]
    expected_pct = (pred - current) / current

    if expected_pct >= 0.0025:
        direction = "long"
        raw_pct = (future - current) / current
        pnl_pct = max(raw_pct, -stop_loss_pct)
    elif expected_pct <= -0.0025:
        direction = "short"
        raw_pct = (current - future) / current
        pnl_pct = max(raw_pct, -stop_loss_pct)
    else:
        continue

    is_liquidated = pnl_pct == -stop_loss_pct
    pnl = pnl_pct * position_size * leverage
    balance += pnl
    results.append([
        timestamps[i],
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

# 4. Analyze liquidations, losses, and profit distribution
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

# 수익률 분포 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.hist(result_df["pnl(%)"], bins=30, color="skyblue", edgecolor="black")
plt.title("수익률 분포 Histogram")
plt.xlabel("pnl(%)")
plt.ylabel("빈도")
plt.grid(True)
plt.tight_layout()
plt.show()

# 수익 곡선 (잔고 변화)
plt.figure(figsize=(10, 4))
plt.plot(result_df["balance"], label="Balance", color="green")
plt.title("잔고 추이 (Balance Over Time)")
plt.xlabel("거래 순서")
plt.ylabel("잔고 ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

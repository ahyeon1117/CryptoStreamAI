import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import build_lstm_model, create_sequences, scaler
from project.collector import get_binance_ohlcv
import tensorflow as tf

# ✅ 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 1. 데이터 준비 (한 달치 데이터)
df = get_binance_ohlcv(days=14)
train_df = df.iloc[:-1440*7]  # 최근 7일 전까지

# 최근 7일을 테스트로 사용
test_df = df.iloc[-1440*7:]

# ✅ 2. 모델 학습
train_prices = train_df['close'].values.reshape(-1, 1)
scaled = scaler.fit_transform(train_prices)
X, y = create_sequences(scaled)

# 개선된 모델 구조 적용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mae')
model.fit(X, y, epochs=30, batch_size=512)

# ✅ 3. 테스트 및 시뮬레이션 수행
initial_balance = 1000.0
balance = initial_balance
position_size = 100  # USD
leverage = 15
stop_loss_pct = 0.01
take_profit_pct = 0.05
entry_threshold = 0.002
fee_rate = 0.001

# 테스트 데이터 준비
test_prices = test_df['close'].values.reshape(-1, 1)
timestamps = test_df['timestamp'].values
scaled_test = scaler.transform(test_prices)
input_seqs = []

for i in range(60, len(test_prices) - 15):
    input_seqs.append(scaled_test[i - 60:i])

input_seqs = np.array(input_seqs)
preds = model.predict(input_seqs, verbose=0)

results = []
entry_times = []
exit_times = []

for i in tqdm(range(len(input_seqs)), desc="Simulating", unit="step"):
    current = float(test_prices[i][0])
    future = float(test_prices[i+15][0])  # 15분 후
    pred = preds[i][0]
    pred = scaler.inverse_transform([[pred]])[0][0]

    expected_pct = ((pred - current) / current) * 2

    if i < 60:
        continue

    ma_60 = np.mean(test_prices[i-60:i])
    ma_10 = np.mean(test_prices[i-10:i])

    if expected_pct >= entry_threshold and current > ma_60 and ma_10 > ma_60:
        direction = "long"
        raw_pct = (future - current) / current
        pnl_pct = max(min(raw_pct, take_profit_pct), -stop_loss_pct)
    else:
        continue

    pnl_pct_after_fee = pnl_pct - fee_rate
    is_liquidated = pnl_pct_after_fee <= -stop_loss_pct
    pnl = pnl_pct_after_fee * position_size * leverage
    balance += pnl

    entry_times.append(pd.to_datetime(timestamps[i]))
    exit_times.append(pd.to_datetime(timestamps[i + 15]))

    results.append([
        timestamps[i],
        current,
        future,
        pnl_pct_after_fee * 100 * leverage,
        direction,
        pnl > 0,
        round(balance, 2),
        is_liquidated
    ])

# ✅ 4. 결과 저장 및 분석
with open("sim_result.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "entry", "exit", "pnl(%)", "direction", "hit", "balance", "liquidated"])
    writer.writerows(results)

result_df = pd.read_csv("sim_result.csv")
print(f"❌ 청산된 횟수: {int(result_df['liquidated'].sum())}회")

loss_trades = result_df[result_df["pnl(%)"] < 0]
total_loss = loss_trades["pnl(%)"].sum()
avg_loss = loss_trades["pnl(%)"].mean()
print(f"💸 총 손실률 합계: {total_loss:.2f}%")
print(f"📉 평균 손실률: {avg_loss:.2f}%")

num_wins = result_df["hit"].sum()
num_losses = len(result_df) - num_wins
print(f"✅ 수익 낸 거래 수: {int(num_wins)}회")
print(f"❌ 손실 낸 거래 수: {int(num_losses)}회")

long_df = result_df[result_df["direction"] == "long"]
print(f"🔺 롱 수익률 합계: {long_df['pnl(%)'].sum():.2f}%")

# 💀 청산 확률 계산
num_liquidated = result_df[result_df['liquidated']].shape[0]
total_trades = len(result_df)
print(f"💀 청산 확률: {(num_liquidated / total_trades * 100):.2f}%" if total_trades else "💀 청산 확률: 계산 불가")

# 📊 평균 보유 시간
if entry_times and exit_times:
    holding_durations = [(exit - entry).total_seconds() / 60 for entry, exit in zip(entry_times, exit_times)]
    avg_holding_time = sum(holding_durations) / len(holding_durations)
    print(f"⏱️ 평균 포지션 보유 시간: {avg_holding_time:.2f}분")

# 시각화
plt.figure(figsize=(10, 4))
plt.hist(result_df["pnl(%)"], bins=30, color="skyblue", edgecolor="black")
plt.title("수익률 분포 Histogram")
plt.xlabel("pnl(%)")
plt.ylabel("빈도")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(result_df["balance"], label="Balance", color="green")
plt.title("잔고 추이 (Balance Over Time)")
plt.xlabel("거래 순서")
plt.ylabel("잔고 ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

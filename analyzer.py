import pandas as pd

df = pd.read_csv("sim_result.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 투자 시점 간 차이 (진입 시각 → 다음 진입 시각 기준)
df['next_timestamp'] = df['timestamp'].shift(-1)
df['holding_minutes'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds() / 60

print("총 포지션 수:", len(df))
print("승률:", df["hit"].mean() * 100)
print("평균 수익률:", df["pnl(%)"].mean())
print("누적 수익률:", df["pnl(%)"].sum())
print("최종 잔고:", df["balance"].iloc[-1])
print("수익률(%):", (df["balance"].iloc[-1] - 1000) / 1000 * 100)
print("평균 투자 주기 (분):", df["holding_minutes"].mean())
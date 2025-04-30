import pandas as pd

df = pd.read_csv("sim_result.csv")
print("총 포지션 수:", len(df))
print("승률:", df["hit"].mean() * 100)
print("평균 수익률:", df["pnl(%)"].mean())
print("누적 수익률:", df["pnl(%)"].sum())
print("최종 잔고:", df["balance"].iloc[-1])
print("수익률(%):", (df["balance"].iloc[-1] - 1000) / 1000 * 100)

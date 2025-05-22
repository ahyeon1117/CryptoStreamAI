import threading
import time
import pandas as pd
import ta
import tkinter as tk
from tkinter import ttk
import itertools

from ohlcv import fetch_5min_ohlcv
from top_volume import get_top_100_futures_coins

# 볼린저 신호
def bb_reversal_signal(df, rsi_long, rsi_short, vol_ratio):
    if len(df) < 25:
        return None
    close = df['Close']
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    bb = ta.volatility.BollingerBands(close, window=20)
    lower_band = bb.bollinger_lband().iloc[-1]
    upper_band = bb.bollinger_hband().iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_close = close.iloc[-1]
    recent_vol = df['Volume'].iloc[-10:-1].mean()
    vol_cond = (df['Volume'].iloc[-3:] > recent_vol * vol_ratio).any()
    if last_close < lower_band and last_rsi < rsi_long and vol_cond:
        return 'long'
    if last_close > upper_band and last_rsi > rsi_short and vol_cond:
        return 'short'
    return None

# 그리드
TP_GRID = [0.01, 0.015, 0.02, 0.025]
SL_GRID = [0.007, 0.01, 0.013, 0.016]
HOLD_GRID = [24, 36, 48]
RSI_LONG_GRID = [28, 32, 35]
RSI_SHORT_GRID = [65, 68, 72]
VOL_RATIO_GRID = [1.2, 1.4]

def find_best_signal(df):
    best = None
    best_equity = -999
    for tp, sl, hold, rsi_long, rsi_short, vol_ratio in itertools.product(
        TP_GRID, SL_GRID, HOLD_GRID, RSI_LONG_GRID, RSI_SHORT_GRID, VOL_RATIO_GRID
    ):
        signal = bb_reversal_signal(df, rsi_long, rsi_short, vol_ratio)
        if not signal:
            continue
        entry = df['Close'].iloc[-1]
        if signal == 'long':
            tp_val = entry * (1 + tp)
            sl_val = entry * (1 - sl)
        else:
            tp_val = entry * (1 - tp)
            sl_val = entry * (1 + sl)
        # 백테스트(과거 200봉)
        rets = []
        equity = 1.0
        for i in range(len(df) - 200, len(df) - hold - 1):
            sub = df.iloc[i-25:i]
            sig = bb_reversal_signal(sub, rsi_long, rsi_short, vol_ratio)
            if not sig:
                continue
            ent = df['Close'].iloc[i]
            direction = sig
            fut = df.iloc[i:i+hold+1]
            ex_price = ent
            for _, row in fut.iterrows():
                price = row['Close']
                if direction == 'long':
                    if price >= ent * (1 + tp):
                        ex_price = ent * (1 + tp)
                        break
                    elif price <= ent * (1 - sl):
                        ex_price = ent * (1 - sl)
                        break
                else:
                    if price <= ent * (1 - tp):
                        ex_price = ent * (1 - tp)
                        break
                    elif price >= ent * (1 + sl):
                        ex_price = ent * (1 + sl)
                        break
            else:
                ex_price = fut['Close'].iloc[-1]
            ret = (ex_price / ent - 1) if direction == 'long' else (ent / ex_price - 1)
            rets.append(ret)
            equity *= (1 + ret)
        if len(rets) < 5:
            continue
        avg_ret = sum(rets)/len(rets)
        if equity > best_equity:
            best_equity = equity
            best = dict(
                signal=signal,
                tp=tp, sl=sl, hold=hold,
                rsi_long=rsi_long, rsi_short=rsi_short, vol_ratio=vol_ratio,
                entry=entry,
                avg_ret=avg_ret,
                equity=equity
            )
    return best

active_recommendations = {}
recommend_time_map = {}
COOLTIME_SEC = 60

def recommendation_loop():
    while True:
        now = time.time()
        top_100 = get_top_100_futures_coins()
        scored_list = []
        for symbol, _ in top_100:
            if symbol in recommend_time_map and now - recommend_time_map[symbol] < COOLTIME_SEC:
                continue
            try:
                df = fetch_5min_ohlcv(symbol)
                best = find_best_signal(df)
                if best:
                    scored_list.append((best['equity'], symbol, best))
            except Exception as e:
                print(f"⚠️ {symbol} 처리 중 오류: {e}")
            time.sleep(0.04)
        scored_list.sort(reverse=True)  # equity순
        for _, symbol, best in scored_list:
            active_recommendations[symbol] = {
                'symbol': symbol,
                'direction': best['signal'],
                'entry': best['entry'],
                'tp': best['entry'] * (1 + best['tp']) if best['signal'] == 'long' else best['entry'] * (1 - best['tp']),
                'sl': best['entry'] * (1 - best['sl']) if best['signal'] == 'long' else best['entry'] * (1 + best['sl']),
                'equity': best['equity'],
                'avg_ret': best['avg_ret'],
                'status': 'open',
                'id': symbol
            }
            recommend_time_map[symbol] = now
            print(f"\n✅ 추천: {symbol} | {best['signal']} | TP:{best['tp']} SL:{best['sl']} Hold:{best['hold']} Eq:{best['equity']:.3f}")
        time.sleep(30)

def tkinter_ui():
    def refresh_table():
        for i in tree.get_children():
            tree.delete(i)
        sorted_recs = sorted(active_recommendations.values(), key=lambda x: x['equity'], reverse=True)
        for info in sorted_recs:
            tree.insert('', 'end', iid=info['id'], values=(
                info['symbol'], info['direction'],
                f"{info['entry']:.4f}",
                f"{info['tp']:.4f}",
                f"{info['sl']:.4f}",
                f"{info['equity']:.3f}",
                f"{info['avg_ret']:.4f}",
                info['status']
            ))
        root.after(2000, refresh_table)

    root = tk.Tk()
    root.title("최적 파라미터 실시간 자동 추천")
    # 주요 정보만 컬럼 축소!
    columns = ('코인', '방향', '진입가', 'TP', 'SL', '복리', '평균수익', '상태')
    tree = ttk.Treeview(
        root,
        columns=columns,
        show='headings'
    )
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor='center')
    # 수평 스크롤
    xscrollbar = tk.Scrollbar(root, orient='horizontal', command=tree.xview)
    tree.configure(xscrollcommand=xscrollbar.set)
    tree.pack(expand=True, fill='both')
    xscrollbar.pack(fill='x')
    refresh_table()
    root.mainloop()

if __name__ == '__main__':
    threading.Thread(target=recommendation_loop, daemon=True).start()
    tkinter_ui()

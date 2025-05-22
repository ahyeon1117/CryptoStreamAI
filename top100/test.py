import pandas as pd
import itertools
from ohlcv import fetch_5min_ohlcv
from top_volume import get_top_100_futures_coins
import ta

# 볼린저반전 신호
def bb_reversal_signal(df, rsi_long=35, rsi_short=65, vol_ratio=1.3):
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

# 개별 코인 그리드서치 (코인, TP/SL/Hold 그리드, 신호조건)
def run_bb_gridsearch(symbol, 
                     tp_grid=[0.01, 0.015, 0.02, 0.025], 
                     sl_grid=[0.007, 0.01, 0.013, 0.016],
                     hold_grid=[24, 36, 48],
                     fee_rate=0.0004,
                     rsi_long=32, rsi_short=68, vol_ratio=1.4,
                     max_bars=500):
    try:
        df = fetch_5min_ohlcv(symbol)
        if len(df) < 50:
            return None
    except Exception as e:
        print(f"[{symbol}] 데이터 로딩 실패: {e}")
        return None

    grid_results = []
    for tp, sl, hold_bars in itertools.product(tp_grid, sl_grid, hold_grid):
        results = []
        equity = 1.0
        for i in range(25, min(len(df)-hold_bars, max_bars)):
            window = df.iloc[:i]
            signal = bb_reversal_signal(window, rsi_long, rsi_short, vol_ratio)
            if signal:
                entry = df['Close'].iloc[i]
                direction = signal
                if direction == 'long':
                    tp_val = entry * (1 + tp)
                    sl_val = entry * (1 - sl)
                else:
                    tp_val = entry * (1 - tp)
                    sl_val = entry * (1 + sl)
                future = df.iloc[i:i+hold_bars+1]
                exit_price = entry
                exit_type = "hold"
                for j, row in future.iterrows():
                    price = row['Close']
                    if direction == 'long':
                        if price >= tp_val:
                            exit_price = tp_val
                            exit_type = "tp"
                            break
                        elif price <= sl_val:
                            exit_price = sl_val
                            exit_type = "sl"
                            break
                    else:
                        if price <= tp_val:
                            exit_price = tp_val
                            exit_type = "tp"
                            break
                        elif price >= sl_val:
                            exit_price = sl_val
                            exit_type = "sl"
                            break
                else:
                    exit_price = future['Close'].iloc[-1]
                    exit_type = "hold"
                if direction == 'long':
                    ret = (exit_price / entry) - 1 - 2 * fee_rate
                else:
                    ret = (entry / exit_price) - 1 - 2 * fee_rate
                results.append(ret)
                equity *= (1 + ret)
        # 성과 요약
        if results:
            avg_ret = sum(results)/len(results)
            winrate = sum([r>0 for r in results])/len(results)*100
            grid_results.append({
                'tp': tp,
                'sl': sl,
                'hold_bars': hold_bars,
                'trades': len(results),
                'winrate': winrate,
                'avg_ret': avg_ret*100,
                'equity': equity
            })
    if not grid_results:
        return None
    df_grid = pd.DataFrame(grid_results)
    best = df_grid.sort_values('equity', ascending=False).iloc[0]
    best['symbol'] = symbol
    return best

# 여러 코인 한 번에 최적값 서치
def run_multi_gridsearch(top_n=10, **kwargs):
    top_100 = get_top_100_futures_coins()[:top_n]
    best_results = []
    for symbol, _ in top_100:
        print(f"\n[그리드서치] {symbol} ...")
        best = run_bb_gridsearch(symbol, **kwargs)
        if best is not None:
            best_results.append(best)
    df_best = pd.DataFrame(best_results)
    # 보기 좋게 정렬 및 컬럼 순서 정리
    if not df_best.empty:
        df_best = df_best[['symbol', 'tp', 'sl', 'hold_bars', 'trades', 'winrate', 'avg_ret', 'equity']]
        df_best = df_best.sort_values('equity', ascending=False)
        print("\n===== 코인별 최적 파라미터 및 백테스트 성과 =====")
        print(df_best.to_string(index=False))
    else:
        print("최적값을 찾은 코인이 없습니다.")
    return df_best

# 파라미터 범위 설정
GRID_PARAMS = dict(
    tp_grid=[0.01, 0.015, 0.02, 0.025],  # TP 1~2.5%
    sl_grid=[0.007, 0.01, 0.013, 0.016], # SL 0.7~1.6%
    hold_grid=[24, 36, 48],              # 보유 2~4시간
    rsi_long=32, rsi_short=68,           # 신호 조건
    vol_ratio=1.4,
    max_bars=500
)

# 실행
if __name__ == "__main__":
    run_multi_gridsearch(top_n=10, **GRID_PARAMS)

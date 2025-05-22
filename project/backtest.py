# backtest_tuning.py

import logging
import pandas as pd
from collector import get_binance_ohlcv
from model import ModelManager
from indicators import add_indicators
from config import FEATURES
import numpy as np
from collections import Counter
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────────────────────────────────
def generate_signals(df, mgr, horizon=15, min_prob=0.55, margin=0.2):
    signals = []

    for i in range(60, len(df)):
        df_slice = df.iloc[i-60:i].copy()
        df_slice = add_indicators(df_slice)

        if df_slice.shape[0] < 60:
            signals.append(None)
            continue

        try:
            feats = np.hstack([
                mgr.scalers[f].transform(df_slice[f].values.reshape(-1, 1))
                for f in FEATURES
            ])[None, ...]

            probs = mgr.classifier.predict(feats, verbose=0)[0]
            cls = np.argmax(probs)
            p_down, p_neutral, p_up = probs

            signal = None
            if cls == 2 and p_up > min_prob and (p_up - p_down) > margin:
                signal = 'LONG'
            elif cls == 0 and p_down > min_prob and (p_down - p_up) > margin:
                signal = 'SHORT'
            elif cls == 1:
                signal = 'NEUTRAL'
        except:
            signal = None

        signals.append(signal)

    return signals


def run_backtest(df, signals, exit_pct=0.3):
    in_position = False
    entry_price = entry_time = entry_side = None
    trades = []

    for i in range(len(signals)):
        price = df['close'].iloc[i]
        ts    = df.index[i]
        sig   = signals[i]

        if not in_position:
            if sig in ['LONG', 'SHORT']:
                in_position = True
                entry_price = price
                entry_time  = ts
                entry_side  = sig
        else:
            pnl = (
                (price - entry_price) / entry_price * 100
                if entry_side == 'LONG'
                else (entry_price - price) / entry_price * 100
            )

            if abs(pnl) >= exit_pct or sig == 'NEUTRAL':
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'side': entry_side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl
                })
                in_position = False
                entry_price = entry_time = entry_side = None

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("📥 Loading historical data...")
    df = get_binance_ohlcv(days=7)
    mgr = ModelManager(df, classification_horizon=15)

    prob_range = [0.5, 0.55, 0.6]
    margin_range = [0.1, 0.15, 0.2, 0.25]

    results = []

    logger.info("🔍 Running parameter sweep...")
    for min_prob, margin in product(prob_range, margin_range):
        logger.info(f"▶️ Trying: min_prob={min_prob}, margin={margin}")
        signals = generate_signals(df, mgr, min_prob=min_prob, margin=margin)
        trades = run_backtest(df.iloc[-len(signals):], signals)

        win_rate = (trades['pnl'] > 0).mean() * 100 if not trades.empty else 0
        avg_pnl  = trades['pnl'].mean() if not trades.empty else 0
        max_loss = trades['pnl'].min() if not trades.empty else 0
        count    = len(trades)

        results.append({
            'min_prob': min_prob,
            'margin': margin,
            'trades': count,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'max_loss': max_loss
        })

    df_result = pd.DataFrame(results)
    df_result.to_csv("backtest_tuning_results.csv", index=False)
    logger.info("✅ Tuning results saved: backtest_tuning_results.csv")

    print(df_result.sort_values(by=["win_rate", "avg_pnl"], ascending=False).to_string(index=False))

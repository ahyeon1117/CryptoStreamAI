import pandas as pd
import numpy as np
from config import ATR_PERIOD, WINDOW_SIZE, FEATURES

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['open','high','low']:
        df[col] = df.get(col, df['close'])
    df['volume'] = df.get('volume', 0)

    # 기본 지표
    df['momentum']  = df['close'].diff().fillna(0)
    hl              = df['high'] - df['low']
    hpc             = (df['high'] - df['close'].shift()).abs()
    lpc             = (df['low'] - df['close'].shift()).abs()
    tr              = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['atr']       = tr.rolling(ATR_PERIOD, min_periods=1).mean().bfill()
    df['ema_short'] = df['close'].ewm(span=5).mean()
    df['ema_long']  = df['close'].ewm(span=20).mean()

    # RSI
    delta      = df['close'].diff()
    up         = delta.clip(lower=0)
    down       = -delta.clip(upper=0)
    roll_up    = up.ewm(span=14).mean()
    roll_down  = down.ewm(span=14).mean()
    rs         = roll_up / (roll_down + 1e-8)
    df['rsi']  = 100 - 100 / (1 + rs)

    # 추가 지표 시작 -----------------------------------------------------------

    # 1. 가격 변화율 (최근 WINDOW_SIZE 기준)
    df['price_change_ratio'] = df['close'].pct_change(WINDOW_SIZE).fillna(0)

    # 2. 거래량 변화율 (최근 WINDOW_SIZE 기준)
    df['volume_change_ratio'] = df['volume'].pct_change(WINDOW_SIZE).fillna(0)

    # 3. OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # 4. ROC (Rate of Change)
    df['roc'] = df['close'].pct_change(periods=10).fillna(0)

    # 5. CMO (Chande Momentum Oscillator)
    diff     = df['close'].diff()
    sum_up   = diff.where(diff > 0, 0).rolling(14).sum()
    sum_down = -diff.where(diff < 0, 0).rolling(14).sum()
    df['cmo'] = 100 * (sum_up - sum_down) / (sum_up + sum_down + 1e-8)

    # 6. Trend Strength (ema_long - ema_short)
    df['trend_strength'] = df['ema_long'] - df['ema_short']

    # VWAP
    pv = (df['close'] * df['volume']).rolling(WINDOW_SIZE).sum()
    vv = df['volume'].rolling(WINDOW_SIZE).sum()
    df['vwap'] = pv / (vv + 1e-8)

    # NaN 제거 (필요 피처만)
    return df.dropna(subset=FEATURES)

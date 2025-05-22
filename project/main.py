# main.py

import asyncio
import json
import logging
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import websockets

from config import (
    WINDOW_SIZE, DEFAULT_HIST_DAYS, FEATURES,
    PRED_LOG_PATH, EXIT_PCT_THRESH, BINANCE_WS_URI
)
from collector import get_binance_ohlcv
from model import ModelManager
from indicators import add_indicators
from db_utils import insert_prediction_to_db, insert_trade_entry, insert_trade_exit

# ─── 로깅 설정 ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


async def run_prediction_logger():
    prob_history_len = 300
    prob_buffer = deque(maxlen=prob_history_len)

    while True:
        try:
            logger.info("📥 Loading historical data...")
            df_hist = get_binance_ohlcv(DEFAULT_HIST_DAYS)
            mgr     = ModelManager(df_hist, classification_horizon=15)
            reg     = mgr.reg_models[15]
            logger.info("✅ Model ready (horizon=15)")

            buff             = deque(maxlen=WINDOW_SIZE)
            last_close_time  = None
            smooth_prev      = None
            alpha            = 0.3
            in_position      = False
            entry_price      = None
            entry_time       = None
            entry_side       = None

            logger.info(f"🔗 Connecting to {BINANCE_WS_URI}")
            async with websockets.connect(BINANCE_WS_URI, ping_interval=None) as ws:
                logger.info("🔗 Connected to 1m kline stream")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        k    = data.get('k', {})

                        if not k.get('x', False):
                            continue

                        close_time  = k['t']
                        close_price = float(k['c'])

                        if close_time != last_close_time:
                            logger.info(f"📦 New closed candle @ {close_time}, price={close_price}")
                            buff.append(close_price)
                            last_close_time = close_time

                        if len(buff) < 60:
                            logger.debug(f"⏳ Waiting for 60 candles, current={len(buff)}")
                            continue

                        # ───── 1) Feature 준비 ─────
                        dfw = pd.DataFrame({'close': list(buff)})
                        df2 = add_indicators(dfw)

                        if df2.shape[0] < 60:
                            logger.warning(f"❌ Not enough rows after indicators: {df2.shape[0]}")
                            continue

                        df2_trimmed = df2[-60:]
                        feats = np.hstack([
                            mgr.scalers[f].transform(df2_trimmed[f].values.reshape(-1, 1))
                            for f in FEATURES
                        ])[None, ...]

                        now = datetime.now(timezone.utc).astimezone()

                        # ───── 2) 회귀 예측 ─────
                        p_norm   = reg.predict(feats, verbose=0).flatten()[0]
                        logr     = p_norm * mgr.label_std[15] + mgr.label_mean[15]
                        pred_raw = close_price * np.exp(logr)
                        raw_pct  = (pred_raw - close_price) / close_price * 100

                        # ───── 3) 스무딩 ─────
                        pred_smooth = (
                            pred_raw if smooth_prev is None
                            else alpha * pred_raw + (1 - alpha) * smooth_prev
                        )
                        smooth_prev = pred_smooth
                        smooth_pct  = (pred_smooth - close_price) / close_price * 100

                        # ───── 4) 분류기 예측 (전략 조건 포함) ─────
                        probs = mgr.classifier.predict(feats, verbose=0)[0]
                        cls   = np.argmax(probs)
                        p_down, p_neutral, p_up = probs

                        # 전략적 조건
                        min_prob = 0.55
                        margin   = 0.20

                        signal = ''
                        if cls == 2 and p_up > min_prob and (p_up - p_down) > margin:
                            signal = 'LONG'
                        elif cls == 0 and p_down > min_prob and (p_down - p_up) > margin:
                            signal = 'SHORT'
                        elif cls == 1:
                            signal = 'NEUTRAL'

                        # ───── 5) 로그 출력 ─────
                        logger.info(
                            f"[{now.strftime('%Y-%m-%dT%H:%M:%S')}] "
                            f"Close={close_price:.2f} | Raw={raw_pct:.4f}% | Smooth={smooth_pct:.4f}% | "
                            f"Prob=[DOWN:{p_down:.2f}, NEUT:{p_neutral:.2f}, UP:{p_up:.2f}] | "
                            f"Cls={cls} | Signal={signal or 'None'}"
                        )

                        # ───── 6) 예측 결과 DB 기록 ─────
                        insert_prediction_to_db(
                            now,
                            close_price,
                            round(pred_raw, 2),
                            f"{raw_pct:.4f}%",
                            round(pred_smooth, 2),
                            f"{smooth_pct:.4f}%",
                            f"UP:{p_up:.2f}/DOWN:{p_down:.2f}",
                            signal
                        )

                        # ───── 7) 진입 로직 ─────
                        if signal in ['LONG', 'SHORT'] and not in_position:
                            in_position = True
                            entry_price = close_price
                            entry_time  = now
                            entry_side  = signal
                            insert_trade_entry(entry_time, entry_price, entry_side)
                            logger.info(f"🚀 Entered {signal} @ {entry_price:.2f}")

                        # ───── 8) 청산 로직 ─────
                        if in_position:
                            pnl = (
                                (close_price - entry_price) / entry_price * 100
                                if entry_side == 'LONG'
                                else (entry_price - close_price) / entry_price * 100
                            )
                            if abs(pnl) >= EXIT_PCT_THRESH or signal == 'NEUTRAL':
                                insert_trade_exit(
                                    entry_time, entry_price,
                                    now, close_price,
                                    entry_side, pnl
                                )
                                logger.info(
                                    f"💸 Exited {entry_side} @ {close_price:.2f}, PnL={pnl:.2f}%"
                                )
                                in_position = False
                                entry_price = entry_time = entry_side = None

                    except Exception as e:
                        logger.error(f"❌ Prediction block error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            logger.warning("🔄 WebSocket closed, reconnecting in 5s")
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            import traceback
            traceback.print_exc()
            logger.info("🔄 Reconnecting in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(run_prediction_logger())

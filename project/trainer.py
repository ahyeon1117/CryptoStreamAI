import psycopg2
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from model import ModelManager
from indicators import add_indicators
from config import FEATURES, HORIZONS

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "coin",
    "user": "postgres",
    "password": "post123!"
}

# ─────────────────────────────
# DB에서 예측 데이터 불러오기
# ─────────────────────────────
def load_prediction_data_from_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
            SELECT timestamp, price
            FROM prediction_logs
            ORDER BY timestamp
        """
        df = pd.read_sql(query, conn, parse_dates=['timestamp'])
        conn.close()
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load prediction data from DB: {e}")
        return pd.DataFrame()

# ─────────────────────────────
# 모델 재학습 함수
# ─────────────────────────────
def train_models_from_db():
    logging.info(f"[{datetime.now()}] 🔁 모델 재학습 시작 (DB 기반)")
    df = load_prediction_data_from_db()
    if df.empty:
        logging.warning("📭 예측 데이터가 없어 학습을 건너뜁니다.")
        return

    # 가격 컬럼 이름 정리
    df = df.rename(columns={"price": "close"})

    # 기술 지표 추가
    df = add_indicators(df)
    df.dropna(subset=FEATURES, inplace=True)

    # — 여기를 주석 처리하거나 제거 —
    # 기존: 작은 변동만 학습
    # df['log_ret'] = (df['close'].shift(-horizon) / df['close']).apply(lambda x: pd.NA if x <= 0 else np.log(x))
    # df = df[df['log_ret'].abs() > 0.002]
    # df.dropna(inplace=True)

    # 필터 제거 후 바로 학습
    mgr = ModelManager(df, classification_horizon=HORIZONS)
    std = mgr.label_std.get(HORIZONS)
    logging.info(f"[H{HORIZONS}] label_std after retrain: {std:.6f}")

    logging.info(f"[{datetime.now()}] ✅ 모델 재학습 완료")
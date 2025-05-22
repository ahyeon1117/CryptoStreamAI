import psycopg2
import logging
import pandas as pd

# PostgreSQL connection
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "coin",
    "user":     "postgres",
    "password": "post123!"
}

def load_prediction_data_from_db():
    """prediction_logs 테이블에서 timestamp, price 불러오기"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql(
            "SELECT timestamp, price FROM prediction_logs ORDER BY timestamp",
            conn, parse_dates=['timestamp']
        )
        conn.close()
        return df
    except Exception as e:
        logging.error(f"❌ Failed to load prediction data: {e}")
        return pd.DataFrame()

def insert_prediction_to_db(ts, price, pred_raw, raw_pct, pred_smooth, smooth_pct, prob, signal):
    """prediction_logs에 예측 결과 저장"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_logs
                (timestamp, price, pred_raw, raw_pct,
                 pred_smooth, smooth_pct, prob, signal)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (ts, price, pred_raw, raw_pct, pred_smooth, smooth_pct, prob, signal))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"❌ Prediction insert error: {e}")

def insert_trade_entry(entry_time, entry_price, direction):
    """trade_logs에 진입 기록"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_logs (entry_time, entry_price, direction)
            VALUES (%s,%s,%s)
        """, (entry_time, entry_price, direction))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"❌ Trade entry error: {e}")

def insert_trade_exit(entry_time, entry_price, exit_time, exit_price, direction, profit_pct):
    """trade_logs에 청산 기록"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            UPDATE trade_logs
            SET exit_time = %s, exit_price = %s, profit_pct = %s
            WHERE entry_time=%s AND entry_price=%s AND direction=%s AND exit_time IS NULL
        """, (exit_time, exit_price, profit_pct, entry_time, entry_price, direction))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"❌ Trade exit error: {e}")

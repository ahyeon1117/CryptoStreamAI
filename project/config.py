import os
import logging

# ── Data & Window Settings ─────────────────────────────────────────────
WINDOW_SIZE             = 60
ATR_PERIOD              = 14
HORIZONS                = [1, 5, 15, 30]
FEATURES = [
    'close',
    'momentum',
    'ema_short',
    'ema_long',
    'rsi',
    'atr',
    'price_change_ratio',
    'volume_change_ratio',
    'obv',
    'roc',
    'cmo',
    'trend_strength'
]

# ── Entry / Class Threshold ─────────────────────────────────────────────
ENTRY_THRESHOLD_RATIO   = 0.005  # 0.3%
CLASS_THRESHOLD_RATIO   = 0.005  # 0.3%
EXIT_PCT_THRESH = 0.01

# ── Training Hyperparameters ────────────────────────────────────────────
EPOCHS                  = 20
BATCH_SIZE              = 128

# ── Model Checkpoint Paths ──────────────────────────────────────────────
REG_MODEL_PATH          = os.getenv('REG_MODEL_PATH', 'models/reg_model_h{horizon}.weights.h5')
CLS_MODEL_PATH          = os.getenv('CLS_MODEL_PATH', 'models/classifier.weights.h5')

# ── Prediction Logger Settings ─────────────────────────────────────────
PRED_LOG_PATH           = os.getenv('PRED_LOG_PATH', 'predictions.csv')

# ── Binance Data Collector ─────────────────────────────────────────────
DEFAULT_HIST_DAYS       = 120
BINANCE_WS_URI          = os.getenv(
    'BINANCE_WS_URI',
    'wss://stream.binance.com:9443/ws/btcusdt@kline_1m'
)

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL               = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# main.py

import os
import sys
import asyncio
import websockets
import websockets.exceptions
import json
import numpy as np
import csv
import socket
from datetime import datetime, timezone
from lstm_model import create_sequences, scaler, build_lstm_model
from collector import get_binance_ohlcv
import tensorflow as tf
import platform

# ───────────────────────────────────────────────────────────────────────────────
# 상수 설정 (주석으로 역할 설명)
WINDOW_SIZE          = 60               # LSTM 입력 시퀀스 길이
SHORT_MA_WINDOW      = 5                # 단기 MA 기간
LONG_MA_WINDOW       = 20               # 장기 MA 기간
LEVERAGE             = 15               # 가정 레버리지 배율
STOP_LOSS_PCT        = 0.01             # 손절 기준 (1%)
TAKE_PROFIT_PCT      = 0.03             # 익절 기준 (3%)
ENTRY_THRESHOLD_BASE = 0.0002           # 기본 진입 임계 (0.02%)
FEE_RATE             = 0.001            # 수수료 (0.1%)
MAX_HOLDING_MINUTES  = 120              # 최대 보유 시간 (분)
EPOCHS               = 30               # 학습 에포크
BATCH_SIZE           = 512              # 학습 배치 크기
LOG_PATH             = "trade_log.csv"   # 로그 파일 경로
MODEL_PATH           = "lstm_model.h5"   # 모델 가중치 파일 경로
RECONNECT_DELAY      = 5                # 재연결 대기 시간 (초)
PING_INTERVAL_SEC    = 20               # 수동 ping 주기 (초)
# ───────────────────────────────────────────────────────────────────────────────

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

print("✅ GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

def train_model():
    df = get_binance_ohlcv(days=90)
    prices = df['close'].values.reshape(-1, 1)
    scaled = scaler.fit_transform(prices)
    X, y = create_sequences(scaled)
    seq_len = X.shape[1]
    model = build_lstm_model(input_shape=(seq_len, 1))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(MODEL_PATH)
    return model

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ 모델 파일이 없습니다. 학습을 수행합니다.")
        train_model()
    model = build_lstm_model(input_shape=(WINDOW_SIZE, 1))
    model.load_weights(MODEL_PATH)
    df_hist = get_binance_ohlcv(days=90)
    scaler.fit(df_hist['close'].values.reshape(-1, 1))
    return model

async def heartbeat(ws):
    try:
        while True:
            await ws.ping()
            await asyncio.sleep(PING_INTERVAL_SEC)
    except asyncio.CancelledError:
        return

async def run_bot():
    model = load_model()
    seq = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

    while True:
        try:
            async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as ws:
                # TCP Keepalive 설정
                transport = ws.transport
                sock = transport.get_extra_info('socket')
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

                ping_task = asyncio.create_task(heartbeat(ws))

                # 로그 파일 열기, UTF-8 BOM 포함
                with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as log_file:
                    log_writer = csv.writer(log_file)
                    # 헤더: 타임스탬프, 거래유형, 현재가, 예측가, 수익률
                    log_writer.writerow(["타임스탬프", "거래유형", "현재가", "예측가", "수익률"])

                    print("🚀 연결 성공, 봇 시작")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        price = float(data['c'])
                        ts = datetime.now(timezone.utc).isoformat()

                        # 시퀀스 업데이트
                        seq.append(price)
                        if len(seq) > WINDOW_SIZE:
                            seq.pop(0)
                        if len(seq) < WINDOW_SIZE:
                            continue

                        # 기술 지표
                        short_ma  = np.mean(seq[-SHORT_MA_WINDOW:])
                        long_ma   = np.mean(seq[-LONG_MA_WINDOW:])
                        price_std = np.std(seq[-LONG_MA_WINDOW:])

                        # 예측
                        scaled_seq  = scaler.transform(np.array(seq).reshape(-1, 1))[np.newaxis, ...]
                        pred_scaled = model.predict(scaled_seq, verbose=0)[0][0]
                        pred        = scaler.inverse_transform([[pred_scaled]])[0][0]
                        expected_pct = (pred - price) / price

                        # 동적 임계값 및 모멘텀
                        vol_pct          = price_std / price
                        dyn_entry_thresh = ENTRY_THRESHOLD_BASE + vol_pct * 0.003
                        momentum         = price - seq[-2]

                        # 디버그 로그
                        print(f"[디버그] 예상수익률={expected_pct*100:.2f}% | 동적임계값={dyn_entry_thresh*100:.2f}% | 모멘텀={momentum:.2f}")

                        # 진입 로직
                        if not in_position:
                            if expected_pct > dyn_entry_thresh and momentum > 0:
                                in_position = True
                                entry_price, entry_time = price, datetime.now(timezone.utc)
                                print(f"✅ STRONG ENTRY @ {entry_price:.2f}")
                                log_writer.writerow([ts, "강력 진입", price, round(pred,2), ""])
                            elif expected_pct > ENTRY_THRESHOLD_BASE and short_ma > long_ma:
                                in_position = True
                                entry_price, entry_time = price, datetime.now(timezone.utc)
                                print(f"✅ WEAK ENTRY @ {entry_price:.2f}")
                                log_writer.writerow([ts, "보조 진입", price, round(pred,2), ""])
                            elif momentum > 0:
                                in_position = True
                                entry_price, entry_time = price, datetime.now(timezone.utc)
                                print(f"✅ MOMENTUM ENTRY @ {entry_price:.2f}")
                                log_writer.writerow([ts, "모멘텀 진입", price, round(pred,2), ""])
                            else:
                                print("🚫 진입 없음")
                        else:
                            # 청산 로직
                            pnl = (price - entry_price) / entry_price - FEE_RATE
                            hold_min = (datetime.now(timezone.utc) - entry_time).total_seconds()/60
                            typ = None
                            if hold_min > MAX_HOLDING_MINUTES:
                                typ = "시간초과"
                            elif pnl <= -STOP_LOSS_PCT:
                                typ = "손절"
                            elif pnl >= TAKE_PROFIT_PCT:
                                typ = "익절"
                            if typ:
                                print(f"🔔 청산({typ}) @ {price:.2f}, 수익률={pnl*100:.2f}%")
                                log_writer.writerow([ts, typ, price, round(pred,2), round(pnl*100,2)])
                                in_position = False

                        log_file.flush()
                ping_task.cancel()

        except Exception as e:
            print("⚠️ 연결 오류, 재시도 중:", e)
            await asyncio.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    asyncio.run(run_bot())


from collector import get_binance_ohlcv
from lstm_model import build_lstm_model, create_sequences, scaler
from realtime import realtime_predict
import numpy as np
import asyncio
import platform

# 1. Windows 안정성 확보 (이벤트 루프 정책 설정)
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 2. 데이터 수집 + 모델 학습
df = get_binance_ohlcv(days=7)
prices = df['close'].values.reshape(-1, 1)
scaled = scaler.fit_transform(prices)
X, y = create_sequences(scaled)

model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=3, batch_size=32)

# 3. 실시간 예측 시작
asyncio.run(realtime_predict(model, scaler))
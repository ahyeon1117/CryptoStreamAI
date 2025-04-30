from collector import get_binance_ohlcv
from lstm_model import build_lstm_model, create_sequences, scaler
import asyncio
from realtime import realtime_predict

# 데이터 수집
df = get_binance_ohlcv()
data = df['close'].values.reshape(-1, 1)
scaled_data = scaler.fit_transform(data)

# 시퀀스 생성 및 모델 학습
X, y = create_sequences(scaled_data)
model = build_lstm_model((X.shape[1], 1))
model.fit(X, y, epochs=3, batch_size=32)

# 실시간 예측 실행
asyncio.run(realtime_predict(model))

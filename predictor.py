import numpy as np

def predict_next_price(prices: list[float]) -> float:
    # 간단한 예측 (EMA + 상승 추세 반영)
    if len(prices) < 10:
        return prices[-1]
    
    ema = np.mean(prices[-10:])
    momentum = prices[-1] - prices[-4]  # 최근 추세
    predicted = prices[-1] + 0.5 * momentum
    return predicted
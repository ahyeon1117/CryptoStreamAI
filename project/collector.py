# collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Binance REST API를 통해 과거 OHLCV(캔들) 데이터를 일정 기간(days) 단위로 분할 조회
# 반환: timestamp(UTC), open, high, low, close, volume 형태의 DataFrame

def get_binance_ohlcv(days: int, symbol: str = 'BTCUSDT', interval: str = '1m') -> pd.DataFrame:
    # UTC timezone-aware datetime 사용
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    limit = 1000  # 바이낸스 최대 조회 개수 제한

    while True:
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        # 데이터가 리스트 형태인지 확인
        if not isinstance(data, list):
            raise TypeError(f"Unexpected response format: {type(data)}")
        if not data:
            break
        # OHLCV 항목만 추출
        for entry in data:
            # [0]=timestamp, [1]=open, [2]=high, [3]=low, [4]=close, [5]=volume
            all_data.append({
                'timestamp': entry[0],
                'open': float(entry[1]),
                'high': float(entry[2]),
                'low': float(entry[3]),
                'close': float(entry[4]),
                'volume': float(entry[5])
            })
        # 다음 조회 구간 설정: 마지막 timestamp 이후
        last_ts = data[-1][0]
        if last_ts >= end_time or len(data) < limit:
            break
        start_time = last_ts + 1

    # DataFrame 생성 및 timestamp 변환
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df
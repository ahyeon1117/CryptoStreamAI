import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(path, horizon, save):
    df = pd.read_csv(
        path, skiprows=1,
        names=['timestamp','price','pred_raw','raw_pct',
               'pred_smooth','smooth_pct','prob','signal'],
        encoding='utf-8-sig'
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['raw_pct']   = df['raw_pct'].str.rstrip('%').astype(float)
    df['smooth_pct']= df['smooth_pct'].str.rstrip('%').astype(float)

    # 로그 수익률로 변환
    df['raw_log']    = np.log1p(df['raw_pct']/100)
    df['smooth_log'] = np.log1p(df['smooth_pct']/100)

    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'], df['raw_log'],    label='Raw Log-Return')
    plt.plot(df['timestamp'], df['smooth_log'], '--', label='Smoothed Log-Return')
    plt.axhline(np.log1p(0.004), color='gray', linestyle=':', label='0.4% Threshold')
    plt.xlabel('Time')
    plt.ylabel('Log-Return')
    plt.title(f'Predicted Log-Returns ({horizon}min ahead)')
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f'pred_logs_{horizon}m.png')
    else:
        plt.show()

if __name__ == '__main__':
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='predictions.csv')
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--save', action='store_true')
    args = p.parse_args()
    main(args.path, args.horizon, args.save)

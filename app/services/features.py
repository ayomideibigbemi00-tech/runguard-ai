from __future__ import annotations
import numpy as np
import pandas as pd
from config import FEATURES, WINDOW_SIZE


def engineer_features(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    df['return_1'] = df['close'].pct_change()
    df['return_3'] = df['close'].pct_change(3)
    df['return_6'] = df['close'].pct_change(6)
    df['sma_7'] = df['close'].rolling(7).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility_7'] = df['return_1'].rolling(7).std()
    df['range_pct'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def make_supervised(candles: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if horizon < 1:
        raise ValueError('horizon must be >= 1')
    df = engineer_features(candles)
    if len(df) < WINDOW_SIZE + horizon + 20:
        raise ValueError('Not enough candle data for this horizon.')

    values = df[FEATURES].to_numpy(dtype=np.float64)
    closes = df['close'].to_numpy(dtype=np.float64)
    x, y, target_close = [], [], []
    for end in range(WINDOW_SIZE, len(df) - horizon):
        x.append(values[end - WINDOW_SIZE:end].reshape(-1))
        y.append((closes[end + horizon] / closes[end]) - 1.0)
        target_close.append(closes[end + horizon])
    return np.asarray(x), np.asarray(y), np.asarray(target_close), df

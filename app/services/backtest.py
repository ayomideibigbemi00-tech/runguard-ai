from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES, HORIZONS
from app.model.network import NeuralNetwork


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from raw OHLCV data."""
    df = df.copy()
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['return_6'] = df['close'].pct_change(6)
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['volatility_7'] = df['close'].pct_change().rolling(window=7).std()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    return df[FEATURES]


def _train_model(df, horizon, test_fraction=0.20, refit_every=24):
    """Train the neural network for a single backtest window."""
    features_df = _engineer_features(df)
    target = df['close'].shift(-horizon).rename('target')

    combined = pd.concat([features_df, target], axis=1)
    combined = combined.dropna()

    X = combined[FEATURES].values.astype(np.float64)
    y = combined['target'].values.astype(np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) < 30:
        raise ValueError("Not enough data")

    split = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Train neural network
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=32, learning_rate=0.01)
    model.train(X_train, y_train, epochs=200)

    # Evaluate
    y_pred_test = np.array([model.predict(x.reshape(1, -1)) for x in X_test])
    y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    mae = np.mean(np.abs(y_pred_test - y_test))
    mae_pct = np.mean(np.abs((y_pred_test - y_test) / y_test)) * 100
    
    # Baseline
    baseline_pred = np.full_like(y_test, y_train[-1])
    baseline_mae = np.mean(np.abs(baseline_pred - y_test))
    baseline_mae_pct = np.mean(np.abs((baseline_pred - y_test) / y_test)) * 100

    return mae, mae_pct, baseline_mae, baseline_mae_pct


def walk_forward_backtest(
    coin_id: str,
    interval: str,
    horizon: int,
    test_fraction: float = 0.20,
    refit_every: int = 24,
    require_real_data: bool = True,
) -> dict:
    """
    Perform a walk-forward backtest on a single coin/interval/horizon.
    Returns metrics for analysis.
    """
    from app.services.data import load_candles
    
    # Load historical data
    df, fallback = load_candles(coin_id, interval, allow_fallback=not require_real_data)
    if df is None or df.empty:
        raise ValueError("No historical data available")

    # Train and evaluate
    mae, mae_pct, baseline_mae, baseline_mae_pct = _train_model(
        df, horizon, test_fraction=test_fraction, refit_every=refit_every
    )

    # Directional accuracy (for this test window)
    features_df = _engineer_features(df)
    target = df['close'].shift(-horizon).rename('target')
    combined = pd.concat([features_df, target], axis=1).dropna()
    
    y_true = combined['target'].values
    X_all = combined[FEATURES].values.astype(np.float64)
    
    if len(X_all) > 30:
        split = int(len(X_all) * (1 - test_fraction))
        y_test_actual = y_true[split:]
        
        # Direction correctness
        directions_true = np.sign(y_test_actual[1:] - y_test_actual[:-1])
        directions_pred = np.sign(y_pred_test[1:] - y_pred_test[:-1]) if len(y_pred_test) > 1 else np.array([0])
        
        if len(directions_pred) < len(directions_true):
            directions_pred = np.pad(directions_pred, (0, len(directions_true) - len(directions_pred)), 'constant')
        
        correct_direction = np.sum(directions_true == directions_pred) / len(directions_true) * 100
    else:
        correct_direction = 0.0

    return {
        'coin_id': coin_id,
        'interval': interval,
        'horizon': horizon,
        'strategy': 'NeuralNetwork',
        'metrics': {
            'mae': float(mae),
            'mae_pct': float(mae_pct),
            'baseline_mae': float(baseline_mae),
            'baseline_mae_pct': float(baseline_mae_pct),
            'directional_accuracy': float(correct_direction),
        },
    }
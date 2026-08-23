from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import json
from pathlib import Path

from config import (
    CACHE_DIR, COIN_MAP, WINDOW_SIZE, FEATURES,
    HORIZONS
)
from app.services.data import (
    load_candles, cache_info, fetch_current_prices, normalize_coin_id
)
from app.services.features import build_features
from app.services.predictions import (
    save_prediction, list_predictions, resolve_due_predictions, prediction_summary
)


class PredictionResult:
    def __init__(
        self,
        predicted_price: float,
        current_price: float,
        change_pct: float,
        direction: str,
        interval: str,
        horizon: int,
        horizon_label: str,
        strategy: str,
        validation_mae_pct: float,
        baseline_validation_mae_pct: float,
        current_price_observed_at_utc: str,
        target_time_utc: str,
    ):
        self.predicted_price = predicted_price
        self.current_price = current_price
        self.change_pct = change_pct
        self.direction = direction
        self.interval = interval
        self.horizon = horizon
        self.horizon_label = horizon_label
        self.strategy = strategy
        self.validation_mae_pct = validation_mae_pct
        self.baseline_validation_mae_pct = baseline_validation_mae_pct
        self.current_price_observed_at_utc = current_price_observed_at_utc
        self.target_time_utc = target_time_utc


# Simple neural network from scratch (for illustration - your actual network is in app/model/network.py)
class SimpleNN:
    def __init__(self, input_size: int, hidden_size: int = 32, lr: float = 0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def predict(self, X):
        return self.forward(X).flatten()[0]


def train_model(coin_id: str, interval: str, horizon: int) -> SimpleNN:
    """
    Loads cached data, builds features, and trains a simple model.
    Uses your existing feature engineering.
    """
    df, fallback = load_candles(coin_id, interval, allow_fallback=False)
    
    # Build features (this is your existing feature engineering)
    features_df = build_features(df)
    
    # Prepare X and y
    X = features_df[FEATURES].values
    y = features_df['close'].shift(-horizon).values  # Predict price 'horizon' steps ahead
    
    # Drop NaN rows
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    if len(X) < WINDOW_SIZE:
        raise ValueError(f"Not enough data for {coin_id}/{interval}/horizon={horizon}")
    
    # Train/test split (chronological)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Train simple model
    model = SimpleNN(input_size=X_train.shape[1])
    for _ in range(200):
        y_pred = model.forward(X_train)
        # Backward pass (gradient descent)
        dZ2 = 2 * (y_pred - y_train.reshape(-1, 1)) / len(y_train)
        dW2 = X_train.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ model.W2.T
        dZ1 = dA1 * (model.z1 > 0)
        dW1 = X_train.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        model.W2 -= model.lr * dW2
        model.b2 -= model.lr * db2
        model.W1 -= model.lr * dW1
        model.b1 -= model.lr * db1
    
    # Validate
    y_pred_test = model.forward(X_test).flatten()
    mae_pct = np.mean(np.abs((y_pred_test - y_test) / y_test)) * 100
    
    # Baseline (recent return)
    baseline_pred = y_test[-1] * (1 + np.mean(np.diff(y_train) / y_train[-20:]))
    baseline_mae_pct = np.mean(np.abs((baseline_pred - y_test) / y_test)) * 100
    
    return model, mean, std, mae_pct, baseline_mae_pct


def predict(coin_id: str, interval: str, horizon: int) -> PredictionResult:
    """
    Makes a live prediction, anchored to current CoinGecko price.
    """
    # Normalize coin ID
    coin_id = normalize_coin_id(coin_id)
    
    # Validate interval/horizon
    if interval not in HORIZONS or horizon not in HORIZONS[interval]:
        raise ValueError("Invalid interval or horizon")
    
    # Get live price
    live_prices = fetch_current_prices()
    if coin_id not in live_prices:
        raise ValueError(f"Live price unavailable for {coin_id}")
    
    current_price = live_prices[coin_id]['price']
    observed_at = live_prices[coin_id]['observed_at_utc']
    
    # Train model
    model, mean, std, validation_mae, baseline_mae = train_model(coin_id, interval, horizon)
    
    # Load latest data and build features for prediction input
    df, _ = load_candles(coin_id, interval, allow_fallback=False)
    features_df = build_features(df)
    latest = features_df.iloc[-1][FEATURES].values.reshape(1, -1)
    
    # Normalize
    latest = (latest - mean) / std
    
    # Predict
    predicted_change = model.predict(latest)
    predicted_price = current_price * (1 + predicted_change / 100)
    
    # Determine direction
    direction = 'UP' if predicted_price > current_price else 'DOWN'
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    # Horizon label
    horizon_label = HORIZONS[interval][horizon]
    
    # Calculate target time
    target_time = datetime.now(timezone.utc)
    if interval == 'hourly':
        target_time = target_time.replace(second=0, microsecond=0) + pd.Timedelta(hours=horizon)
    else:
        target_time = target_time.replace(second=0, microsecond=0) + pd.Timedelta(days=horizon)
    
    # Create result
    result = PredictionResult(
        predicted_price=float(predicted_price),
        current_price=float(current_price),
        change_pct=float(change_pct),
        direction=direction,
        interval=interval,
        horizon=horizon,
        horizon_label=horizon_label,
        strategy='Hybrid',
        validation_mae_pct=float(validation_mae),
        baseline_validation_mae_pct=float(baseline_mae),
        current_price_observed_at_utc=observed_at,
        target_time_utc=target_time.isoformat(),
    )
    
    # Save to history
    save_prediction(
        coin_id=coin_id,
        interval=interval,
        horizon=horizon,
        predicted_price=result.predicted_price,
        current_price=result.current_price,
        horizon_label=result.horizon_label,
        strategy=result.strategy,
        validation_mae_pct=result.validation_mae_pct,
        baseline_validation_mae_pct=result.baseline_validation_mae_pct,
        target_time_utc=result.target_time_utc,
    )
    
    return result
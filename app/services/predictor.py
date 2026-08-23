from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import CACHE_DIR, COIN_MAP, WINDOW_SIZE, FEATURES, HORIZONS
from app.services.data import load_candles, fetch_current_prices, normalize_coin_id
from app.services.predictions import save_prediction


class PredictionResult:
    def __init__(self, predicted_price, current_price, change_pct, direction, interval, horizon, horizon_label,
                 strategy, validation_mae_pct, baseline_validation_mae_pct, current_price_observed_at_utc, target_time_utc):
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


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from raw OHLCV data (no external feature module)."""
    df = df.copy()
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['return_6'] = df['close'].pct_change(6)
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['volatility_7'] = df['close'].pct_change().rolling(window=7).std()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    # Keep only the specified features, drop NaNs
    return df[FEATURES]


class SimpleNN:
    def __init__(self, input_size, hidden_size=32, lr=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def predict(self, X):
        return self.forward(X).flatten()[0]


def _train_model(df, horizon):
    """Train a simple model on the given dataframe and horizon."""
    features = _engineer_features(df)
    X = features.values
    y = df['close'].shift(-horizon).values

    # Drop rows with NaN
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < WINDOW_SIZE:
        raise ValueError("Not enough data")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = SimpleNN(input_size=X_train.shape[1])
    for _ in range(200):
        y_pred = model.forward(X_train)
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

    # Validation
    y_pred_test = model.forward(X_test).flatten()
    mae_pct = np.mean(np.abs((y_pred_test - y_test) / y_test)) * 100

    # Simple baseline: last observed price + average daily return
    last_price = y_train[-1]
    avg_return = np.mean(np.diff(y_train) / y_train[:-1])
    baseline_pred = last_price * (1 + avg_return * horizon)
    baseline_mae_pct = np.mean(np.abs((baseline_pred - y_test) / y_test)) * 100

    return model, mean, std, mae_pct, baseline_mae_pct


def predict(coin_id: str, interval: str, horizon: int) -> PredictionResult:
    coin_id = normalize_coin_id(coin_id)
    if interval not in HORIZONS or horizon not in HORIZONS[interval]:
        raise ValueError("Invalid interval or horizon")

    # Get live price
    live_prices = fetch_current_prices()
    if coin_id not in live_prices:
        raise ValueError("Live price unavailable")
    current_price = live_prices[coin_id]['price']
    observed_at = live_prices[coin_id]['observed_at_utc']

    # Load historical data
    df, fallback = load_candles(coin_id, interval, allow_fallback=False)
    if df is None or df.empty:
        raise ValueError("No historical data")

    # Train model
    model, mean, std, validation_mae, baseline_mae = _train_model(df, horizon)

    # Prepare latest features for prediction
    latest_features = _engineer_features(df).iloc[-1].values.reshape(1, -1)
    latest_features = (latest_features - mean) / std

    # Predict percentage change (scaled) – in this simple model we predict absolute price change ratio
    predicted_change = model.predict(latest_features)
    predicted_price = current_price * (1 + predicted_change)  # Adjust if model outputs price ratio

    direction = 'UP' if predicted_price > current_price else 'DOWN'
    change_pct = ((predicted_price - current_price) / current_price) * 100

    horizon_label = HORIZONS[interval][horizon]
    target_time = datetime.now(timezone.utc)
    if interval == 'hourly':
        target_time = target_time + pd.Timedelta(hours=horizon)
    else:
        target_time = target_time + pd.Timedelta(days=horizon)

    result = PredictionResult(
        predicted_price=float(predicted_price),
        current_price=float(current_price),
        change_pct=float(change_pct),
        direction=direction,
        interval=interval,
        horizon=horizon,
        horizon_label=horizon_label,
        strategy='SimpleNN',
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
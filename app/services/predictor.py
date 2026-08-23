from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import FEATURES, HORIZONS
from app.services.data import load_candles, fetch_current_prices, normalize_coin_id
from app.services.predictions import save_prediction
from app.model.network import NeuralNetwork  # <--- Imports the CORE network


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


def _train_model(df, horizon):
    """
    Trains the neural network.
    Returns the trained model, scaler parameters, and validation metrics.
    """
    features_df = _engineer_features(df)
    target = df['close'].shift(-horizon).rename('target')

    # Combine into one dataframe to ensure perfectly aligned rows (NO NaN mismatch)
    combined = pd.concat([features_df, target], axis=1)
    combined = combined.dropna()

    X = combined[FEATURES].values.astype(np.float64)
    y = combined['target'].values.astype(np.float64)

    # SAFETY GUARD: Replace any 0, negative, or NaN values with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) < 30:
        raise ValueError("Not enough data to train")

    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Normalize features (use training set statistics)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Train Neural Network (Imported from app/model/network.py)
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=32, learning_rate=0.01)
    model.train(X_train, y_train, epochs=200)

    # Evaluate on test set
    y_pred_test = np.array([model.predict(x.reshape(1, -1)) for x in X_test])
    y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=0.0, neginf=0.0)
    mae_pct = np.mean(np.abs((y_pred_test - y_test) / y_test)) * 100

    # Baseline (simple moving average - just for comparison)
    baseline_pred = np.full_like(y_test, y_train[-1])
    baseline_mae_pct = np.mean(np.abs((baseline_pred - y_test) / y_test)) * 100

    return model, mean, std, mae_pct, baseline_mae_pct


def predict(coin_id: str, interval: str, horizon: int, user_id: int) -> PredictionResult:
    """Makes a prediction using the trained neural network and saves it to the user's history."""
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

    # Train Neural Network
    model, mean, std, validation_mae, baseline_mae = _train_model(df, horizon)

    # Prepare the latest feature vector for prediction
    latest_features = _engineer_features(df).iloc[-1].values.reshape(1, -1).astype(np.float64)
    latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
    latest_features = (latest_features - mean) / std

    # Run the neural network forward pass on the latest data
    raw_prediction = model.predict(latest_features)

    # SAFETY GUARD: If the neural network returns NaN or a negative price, fall back to current price
    if np.isnan(raw_prediction) or raw_prediction <= 0:
        raw_prediction = current_price

    predicted_price = float(raw_prediction)

    direction = 'UP' if predicted_price > current_price else 'DOWN'
    change_pct = ((predicted_price - current_price) / current_price) * 100

    horizon_label = HORIZONS[interval][horizon]
    target_time = datetime.now(timezone.utc)
    if interval == 'hourly':
        target_time = target_time + pd.Timedelta(hours=horizon)
    else:
        target_time = target_time + pd.Timedelta(days=horizon)

    result = PredictionResult(
        predicted_price=predicted_price,
        current_price=float(current_price),
        change_pct=float(change_pct),
        direction=direction,
        interval=interval,
        horizon=horizon,
        horizon_label=horizon_label,
        strategy='NeuralNetwork',
        validation_mae_pct=float(validation_mae),
        baseline_validation_mae_pct=float(baseline_mae),
        current_price_observed_at_utc=observed_at,
        target_time_utc=target_time.isoformat(),
    )

    # Save to history (this ties into your user authentication and history page)
    save_prediction(
        user_id=user_id,
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
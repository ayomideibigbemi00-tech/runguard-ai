from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import FEATURES, HORIZONS
from app.services.data import load_candles, fetch_current_prices, normalize_coin_id
from app.services.predictions import save_prediction
from app.model.network import NeuralNetwork


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

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Feature normalization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # SCALE THE TARGET to make MSE loss stable
    target_mean = np.mean(y_train)
    if target_mean == 0:
        target_mean = 1.0
    y_train_scaled = y_train / target_mean
    y_test_scaled = y_test / target_mean

    # Train Neural Network (UNCHANGED CLASS)
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=32, learning_rate=0.01)
    model.train(X_train, y_train_scaled, epochs=200)

    # Validation
    y_pred_test_scaled = np.array([model.predict(x.reshape(1, -1)) for x in X_test])
    y_pred_test_scaled = np.nan_to_num(y_pred_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Unscale the predictions
    y_pred_test = y_pred_test_scaled * target_mean
    y_test_original = y_test_scaled * target_mean
    
    mae_pct = np.mean(np.abs((y_pred_test - y_test_original) / y_test_original)) * 100

    baseline_pred = np.full_like(y_test_original, y_train[-1])
    baseline_mae_pct = np.mean(np.abs((baseline_pred - y_test_original) / y_test_original)) * 100

    return model, mean, std, target_mean, mae_pct, baseline_mae_pct


def predict(coin_id: str, interval: str, horizon: int, user_id: int) -> PredictionResult:
    coin_id = normalize_coin_id(coin_id)
    if interval not in HORIZONS or horizon not in HORIZONS[interval]:
        raise ValueError("Invalid interval or horizon")

    live_prices = fetch_current_prices()
    if coin_id not in live_prices:
        raise ValueError("Live price unavailable")
    current_price = live_prices[coin_id]['price']
    observed_at = live_prices[coin_id]['observed_at_utc']

    df, fallback = load_candles(coin_id, interval, allow_fallback=False)
    if df is None or df.empty:
        raise ValueError("No historical data")

    model, mean, std, target_mean, validation_mae, baseline_mae = _train_model(df, horizon)

    latest_features = _engineer_features(df).iloc[-1].values.reshape(1, -1).astype(np.float64)
    latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
    latest_features = (latest_features - mean) / std

    # Predict scaled value, then unscale
    raw_prediction_scaled = model.predict(latest_features)
    raw_prediction_scaled = np.nan_to_num(raw_prediction_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    predicted_price = raw_prediction_scaled * target_mean

    # CLAMP the prediction so it is never more than 5x or 0.2x the current price
    # This prevents "unrealistic" predictions like Avalanche
    predicted_price = max(min(predicted_price, current_price * 5), current_price * 0.2)

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
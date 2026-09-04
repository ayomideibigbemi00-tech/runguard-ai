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
    # feature engineering: turn raw OHLC candles into model inputs
    df = df.copy()
    df['return_1'] = df['close'].pct_change(1)     # 1-candle % change
    df['return_3'] = df['close'].pct_change(3)      # 3-candle % change
    df['return_6'] = df['close'].pct_change(6)      # 6-candle % change
    df['sma_7'] = df['close'].rolling(window=7).mean()    # 7-period moving average
    df['sma_20'] = df['close'].rolling(window=20).mean()  # 20-period moving average
    df['volatility_7'] = df['close'].pct_change().rolling(window=7).std()  # rolling std of returns
    df['range_pct'] = (df['high'] - df['low']) / df['close']  # candle range as % of price
    return df[FEATURES]


def _train_model(df, horizon):
    features_df = _engineer_features(df)

    # target (y) = % return to the target candle: (future_close / anchor_close) - 1
    # trained on % return, not raw price, so the network's small weights don't explode
    anchor_close = df['close']
    future_close = df['close'].shift(-horizon)
    target = ((future_close / anchor_close) - 1.0).rename('target')

    combined = pd.concat([features_df, target], axis=1)
    combined = combined.dropna()

    X = combined[FEATURES].values.astype(np.float64)
    y = combined['target'].values.astype(np.float64)
    anchor = combined['close'].values.astype(np.float64)  # anchor price per row (for rebuilding price later)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) < 30:
        raise ValueError("Not enough data")

    # chronological 80/20 split (not shuffled, to avoid leaking future data into training)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    anchor_test = anchor[split:]

    # standardize inputs (z-score): X_scaled = (X - mean) / std, fit on train set only
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    X_train = np.clip((X_train - x_mean) / x_std, -5.0, 5.0)   # clip outliers to +/-5 std
    X_test = np.clip((X_test - x_mean) / x_std, -5.0, 5.0)

    # standardize target the same way
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_scaled = (y_train - y_mean) / y_std

    # --- train the neural network (see network.py for forward pass / backprop) ---
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=32, learning_rate=0.01)
    model.train(X_train, y_train_scaled, epochs=200)

    # validate on held-out test set (un-scale predictions back to % return, then price)
    y_pred_test_scaled = np.array([model.predict(x.reshape(1, -1)) for x in X_test])
    y_pred_test_scaled = np.nan_to_num(y_pred_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_test = y_pred_test_scaled * y_std + y_mean   # un-standardize: reverse of (y - mean) / std

    # MAE% = mean absolute error, in price terms
    pred_price = anchor_test * (1.0 + y_pred_test)
    actual_price = anchor_test * (1.0 + y_test)
    mae_pct = np.mean(np.abs((pred_price - actual_price) / actual_price)) * 100

    # naive baseline (predict "no change") for comparison
    baseline_price = anchor_test * (1.0 + y_train[-1])
    baseline_mae_pct = np.mean(np.abs((baseline_price - actual_price) / actual_price)) * 100

    return model, x_mean, x_std, y_mean, y_std, mae_pct, baseline_mae_pct


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

    # train a fresh model live for this request
    model, x_mean, x_std, y_mean, y_std, validation_mae, baseline_mae = _train_model(df, horizon)

    # build + standardize the latest feature row ("right now")
    latest_features = _engineer_features(df).iloc[-1].values.reshape(1, -1).astype(np.float64)
    latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
    latest_features = np.clip((latest_features - x_mean) / x_std, -5.0, 5.0)

    # --- forward pass (inference) ---
    predicted_return_scaled = model.predict(latest_features)
    predicted_return_scaled = np.nan_to_num(predicted_return_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    predicted_return = float(predicted_return_scaled) * y_std + y_mean  # un-standardize

    # safety clip: never let one bad weight update produce a nonsense price
    predicted_return = float(np.clip(predicted_return, -0.9, 5.0))

    # live-anchor: apply predicted % return to the LIVE price, not the last cached candle
    predicted_price = current_price * (1.0 + predicted_return)

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
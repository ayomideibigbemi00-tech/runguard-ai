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

    # Train on the % return to the target candle, not the raw close price.
    # Raw prices (e.g. $78,000) are wildly outside the scale the network's
    # weights are initialized for, so training on them directly makes the
    # gradients explode within a couple of epochs (the network outputs NaN
    # forever after). Returns are small and roughly zero-centered, which
    # this size of network can actually learn.
    anchor_close = df['close']
    future_close = df['close'].shift(-horizon)
    target = ((future_close / anchor_close) - 1.0).rename('target')

    combined = pd.concat([features_df, target], axis=1)
    combined = combined.dropna()

    X = combined[FEATURES].values.astype(np.float64)
    y = combined['target'].values.astype(np.float64)
    # 'close' is already one of FEATURES, so this is just each row's own
    # anchor price - the price the % return was computed relative to.
    anchor = combined['close'].values.astype(np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) < 30:
        raise ValueError("Not enough data")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    anchor_test = anchor[split:]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    X_train = np.clip((X_train - x_mean) / x_std, -5.0, 5.0)
    X_test = np.clip((X_test - x_mean) / x_std, -5.0, 5.0)
    # Clipping matters here specifically: CoinGecko's market_chart endpoint
    # only gives one price per candle, so open/high/low/close collapse to
    # the same value for almost every historical row and 'range_pct' ends
    # up with near-zero variance. A live row where it's briefly nonzero
    # would otherwise standardize into a many-thousand-sigma outlier and
    # blow up the network's output.

    # Standardize the target the same way the inputs are standardized.
    # This is the missing piece that used to make training diverge.
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_scaled = (y_train - y_mean) / y_std

    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=32, learning_rate=0.01)
    model.train(X_train, y_train_scaled, epochs=200)

    y_pred_test_scaled = np.array([model.predict(x.reshape(1, -1)) for x in X_test])
    y_pred_test_scaled = np.nan_to_num(y_pred_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_test = y_pred_test_scaled * y_std + y_mean

    # Report error in price terms (easier to read than raw return numbers),
    # using each test row's own anchor close price to rebuild a price.
    pred_price = anchor_test * (1.0 + y_pred_test)
    actual_price = anchor_test * (1.0 + y_test)
    mae_pct = np.mean(np.abs((pred_price - actual_price) / actual_price)) * 100

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

    model, x_mean, x_std, y_mean, y_std, validation_mae, baseline_mae = _train_model(df, horizon)

    latest_features = _engineer_features(df).iloc[-1].values.reshape(1, -1).astype(np.float64)
    latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=0.0, neginf=0.0)
    latest_features = np.clip((latest_features - x_mean) / x_std, -5.0, 5.0)

    predicted_return_scaled = model.predict(latest_features)
    predicted_return_scaled = np.nan_to_num(predicted_return_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    predicted_return = float(predicted_return_scaled) * y_std + y_mean

    # Safety net only: with a properly scaled target the network shouldn't
    # produce anything like this, but never let one bad weight update turn
    # into a nonsense price on screen (e.g. a coin going to $0 or 100x).
    predicted_return = float(np.clip(predicted_return, -0.9, 5.0))

    # Anchor the prediction to the live price rather than whatever price
    # happened to be last in the historical candle cache - this is the
    # "live-anchored" part of the app.
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
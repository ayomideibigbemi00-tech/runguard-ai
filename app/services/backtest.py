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

    # Train on the % return to the target candle, not the raw close price.
    # Raw prices are wildly outside the scale the network's weights are
    # initialized for, so training on them directly makes the gradients
    # explode within a couple of epochs (the network outputs NaN forever
    # after). Returns are small and roughly zero-centered, which this size
    # of network can actually learn. See predictor.py for the same fix.
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

    split = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    anchor_test = anchor[split:]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    # Clip standardized features: CoinGecko's market_chart endpoint only
    # gives one price per candle, so open/high/low/close collapse to the
    # same value for almost every row and 'range_pct' ends up with
    # near-zero variance. Any row where it's briefly nonzero would
    # otherwise standardize into a huge outlier and destabilize training.
    X_train = np.clip((X_train - mean) / std, -5.0, 5.0)
    X_test = np.clip((X_test - mean) / std, -5.0, 5.0)

    # Standardize the target the same way the inputs are standardized.
    # This is the missing piece that used to make training diverge.
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_scaled = (y_train - y_mean) / y_std

    # Train neural network
    model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=32, learning_rate=0.01)
    model.train(X_train, y_train_scaled, epochs=200)

    # Evaluate
    y_pred_test_scaled = np.array([model.predict(x.reshape(1, -1)) for x in X_test])
    y_pred_test_scaled = np.nan_to_num(y_pred_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred_test = y_pred_test_scaled * y_std + y_mean

    # Report error in price terms (easier to read than raw return numbers),
    # rebuilding a price from each test row's own anchor close price.
    pred_price = anchor_test * (1.0 + y_pred_test)
    actual_price = anchor_test * (1.0 + y_test)

    mae = np.mean(np.abs(pred_price - actual_price))
    mae_pct = np.mean(np.abs((pred_price - actual_price) / actual_price)) * 100

    # Baseline: "tomorrow looks like the most recent known return"
    baseline_price = anchor_test * (1.0 + y_train[-1])
    baseline_mae = np.mean(np.abs(baseline_price - actual_price))
    baseline_mae_pct = np.mean(np.abs((baseline_price - actual_price) / actual_price)) * 100

    # Directional accuracy: did the model get the sign of the move right?
    # (Comparing predicted vs. actual return sign is well-defined per row,
    # unlike comparing consecutive test rows to each other.)
    directional_accuracy = float(np.mean(np.sign(y_pred_test) == np.sign(y_test)) * 100)

    return mae, mae_pct, baseline_mae, baseline_mae_pct, directional_accuracy


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
    mae, mae_pct, baseline_mae, baseline_mae_pct, correct_direction = _train_model(
        df, horizon, test_fraction=test_fraction, refit_every=refit_every
    )

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
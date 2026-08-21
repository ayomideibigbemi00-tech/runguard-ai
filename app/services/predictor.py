from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from config import COIN_MAP, HORIZONS, MODELS_DIR, TRAIN_RATIO, VALIDATION_RATIO, WINDOW_SIZE
from app.model.network import DenseNetwork, StandardScaler
from app.services.data import fetch_current_prices, load_candles, normalize_coin_id
from app.services.features import engineer_features, make_supervised
from app.services.predictions import create_prediction


@dataclass
class PredictionResult:
    interval: str
    horizon: int
    horizon_label: str
    current_price: float
    predicted_price: float
    change: float
    change_pct: float
    direction: str
    source: str
    model_loss: float
    strategy: str
    validation_mae_pct: float
    baseline_validation_mae_pct: float
    prediction_id: str | None = None
    created_at_utc: str | None = None
    target_time_utc: str | None = None
    prediction_status: str = 'pending'
    current_price_observed_at_utc: str | None = None



def _model_key(coin_id: str, interval: str, horizon: int) -> str:
    return hashlib.sha1(f'{coin_id}:{interval}:{horizon}'.encode()).hexdigest()[:16]


def _paths(coin_id: str, interval: str, horizon: int) -> tuple[Path, Path, Path]:
    key = _model_key(coin_id, interval, horizon)
    return MODELS_DIR / f'{key}.npz', MODELS_DIR / f'{key}_x.json', MODELS_DIR / f'{key}_y.json'


def _split(x: np.ndarray, y: np.ndarray):
    n = len(x)
    train_end = max(1, int(n * TRAIN_RATIO))
    val_end = max(train_end + 1, int(n * (TRAIN_RATIO + VALIDATION_RATIO)))
    val_end = min(n - 1, val_end)
    return x[:train_end], y[:train_end], x[train_end:val_end], y[train_end:val_end], x[val_end:], y[val_end:]


def _train(coin_id: str, interval: str, horizon: int):
    candles, fallback = load_candles(coin_id, interval)
    x, y, _, features = make_supervised(candles, horizon)
    xt, yt, xv, yv, _, _ = _split(x, y)

    scaler_x = StandardScaler().fit(xt)
    scaler_y = StandardScaler().fit(yt.reshape(-1, 1))
    xts = scaler_x.transform(xt)
    xvs = scaler_x.transform(xv)
    yts = scaler_y.transform(yt.reshape(-1, 1)).ravel()
    yvs = scaler_y.transform(yv.reshape(-1, 1)).ravel()

    net = DenseNetwork(xts.shape[1], hidden_size=48, seed=42)
    history = net.fit(xts, yts, xvs, yvs, epochs=100, learning_rate=0.0015, batch_size=32, patience=15)

    model_val_scaled = net.predict(xvs)
    model_val = scaler_y.inverse_transform(model_val_scaled.reshape(-1, 1)).ravel()
    baseline_val = np.full(len(yv), float(np.mean(yt[-6:])), dtype=np.float64)
    model_mae = float(np.mean(np.abs(model_val - yv)) * 100.0)
    baseline_mae = float(np.mean(np.abs(baseline_val - yv)) * 100.0)

    # Choose an ensemble weight using validation data only. alpha=1 means pure NN,
    # alpha=0 means the simple recent-return baseline. Because alpha is selected
    # chronologically on validation data, the final test period remains untouched.
    best_alpha = 0.0
    best_mae = baseline_mae
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * model_val + (1.0 - alpha) * baseline_val
        mae = float(np.mean(np.abs(blended - yv)) * 100.0)
        if mae + 1e-12 < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    strategy = 'neural_network' if best_alpha == 1.0 else ('hybrid' if best_alpha > 0.0 else 'recent_return_baseline')
    loss = float(history['val_loss'][-1]) if history['val_loss'] else 0.0

    model_path, sx_path, sy_path = _paths(coin_id, interval, horizon)
    net.save(model_path, {
        'input_size': xts.shape[1], 'hidden_size': 48, 'coin_id': coin_id, 'coin_name': COIN_MAP[coin_id]['name'],
        'interval': interval, 'horizon': horizon, 'window_size': WINDOW_SIZE,
        'feature_count': len(features.columns), 'data_last_timestamp': str(candles['timestamp'].iloc[-1]),
        'data_source': 'synthetic offline fallback' if fallback else 'CoinGecko market data',
        'validation_loss': loss, 'validation_mae_pct': model_mae,
        'baseline_validation_mae_pct': baseline_mae, 'ensemble_alpha': best_alpha,
        'deployment_strategy': strategy,
    })
    sx_path.write_text(json.dumps(scaler_x.to_dict()), encoding='utf-8')
    sy_path.write_text(json.dumps(scaler_y.to_dict()), encoding='utf-8')
    return net, scaler_x, scaler_y, candles, fallback, loss, strategy, model_mae, baseline_mae, best_alpha


def _model_is_current(meta: dict, coin_id: str, interval: str, horizon: int, candles) -> bool:
    return (
        meta.get('coin_id') == coin_id and
        meta.get('interval') == interval and
        int(meta.get('horizon', -1)) == horizon and
        str(meta.get('data_last_timestamp')) == str(candles['timestamp'].iloc[-1])
    )


def predict(coin_id: str, interval: str, horizon: int) -> PredictionResult:
    coin_id = normalize_coin_id(coin_id)
    if horizon not in HORIZONS.get(interval, {}):
        raise ValueError('Unsupported prediction horizon.')

    model_path, sx_path, sy_path = _paths(coin_id, interval, horizon)
    candles, fallback = load_candles(coin_id, interval)
    needs_train = True
    try:
        net, meta = DenseNetwork.load(model_path)
        scaler_x = StandardScaler.from_dict(json.loads(sx_path.read_text(encoding='utf-8')))
        scaler_y = StandardScaler.from_dict(json.loads(sy_path.read_text(encoding='utf-8')))
        needs_train = not _model_is_current(meta, coin_id, interval, horizon, candles)
    except Exception:
        needs_train = True

    if needs_train:
        (net, scaler_x, scaler_y, candles, fallback, validation_loss, strategy, validation_mae, baseline_validation_mae, alpha) = _train(coin_id, interval, horizon)
    else:
        validation_loss = float(meta.get('validation_loss', 0.0))
        strategy = str(meta.get('deployment_strategy', 'neural_network'))
        validation_mae = float(meta.get('validation_mae_pct', 0.0))
        baseline_validation_mae = float(meta.get('baseline_validation_mae_pct', 0.0))
        alpha = float(meta.get('ensemble_alpha', 1.0))

    recent = engineer_features(candles)
    from config import FEATURES
    x_latest = recent[FEATURES].tail(WINDOW_SIZE).to_numpy(dtype=np.float64).reshape(1, -1)
    model_pred_scaled = net.predict(scaler_x.transform(x_latest))[0]
    model_return = float(scaler_y.inverse_transform(np.array([[model_pred_scaled]]))[0, 0])
    recent_returns = recent['return_1'].dropna().tail(6).to_numpy(dtype=np.float64)
    baseline_return = float(np.mean(recent_returns)) if len(recent_returns) else 0.0
    pred_return = alpha * model_return + (1.0 - alpha) * baseline_return

    # The model is trained and evaluated from historical candles, but the
    # monetary prediction MUST be anchored to the actual live market price
    # at the instant the prediction is created. A cached candle close can be
    # hours old and must never masquerade as today's current price.
    live_prices = fetch_current_prices()
    live = live_prices.get(coin_id)
    if not live:
        raise RuntimeError(f'No live price returned for {COIN_MAP[coin_id]["symbol"]}. Prediction not created.')
    current = float(live['price'])
    current_observed_at = str(live.get('observed_at_utc') or '')
    if not current_observed_at:
        raise RuntimeError('Live price response did not include an observation timestamp. Prediction not created.')

    predicted = max(0.0, current * (1 + pred_return))
    change = predicted - current
    change_pct = (change / current * 100) if current else 0.0
    direction = 'Bullish' if change > 0 else 'Bearish' if change < 0 else 'Neutral'
    source = 'Synthetic historical fallback + CoinGecko live price' if fallback else 'Historical CoinGecko data + live CoinGecko price'
    record = create_prediction(
        coin_id=coin_id, interval=interval, horizon=horizon,
        current_price=current, current_price_observed_at_utc=current_observed_at,
        predicted_price=predicted, predicted_change=change, predicted_change_pct=change_pct,
        predicted_direction=direction, source=source,
    )
    return PredictionResult(
        interval, horizon, HORIZONS[interval][horizon], current, predicted, change,
        change_pct, direction, source, validation_loss, strategy, validation_mae,
        baseline_validation_mae, record['prediction_id'], record['created_at_utc'],
        record['target_time_utc'], record['status'], current_observed_at,
    )


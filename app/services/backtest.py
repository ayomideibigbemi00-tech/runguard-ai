from __future__ import annotations
from dataclasses import asdict, dataclass
import time

import numpy as np

from config import COIN_MAP, HORIZONS, WINDOW_SIZE
from app.model.network import DenseNetwork, StandardScaler
from app.services.data import load_candles, normalize_coin_id
from app.services.features import make_supervised


@dataclass
class BacktestMetrics:
    mae_pct: float
    rmse_pct: float
    mape_pct: float
    directional_accuracy_pct: float
    baseline_mae_pct: float
    baseline_rmse_pct: float
    baseline_mape_pct: float
    baseline_directional_accuracy_pct: float
    model_beats_baseline_mae: bool
    model_beats_baseline_rmse: bool
    model_beats_baseline_mape: bool
    model_beats_baseline_directional_accuracy: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float, float]:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)) * 100.0)
    rmse = float(np.sqrt(np.mean(error ** 2)) * 100.0)
    nonzero = np.where(np.abs(actual) > 1e-12, np.abs(actual), np.nan)
    mape = float(np.nanmean(np.abs(error) / nonzero) * 100.0)
    directional = float(np.mean(np.sign(predicted) == np.sign(actual)) * 100.0)
    return mae, rmse, mape, directional


def _baseline_predictions(y_train: np.ndarray, n: int, lookback: int = 6) -> np.ndarray:
    """Recent-return baseline using only observations available before each test point."""
    if len(y_train) == 0:
        return np.zeros(n, dtype=np.float64)
    recent = y_train[-lookback:]
    return np.full(n, float(np.mean(recent)), dtype=np.float64)


def _fit_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[DenseNetwork, StandardScaler, StandardScaler, float]:
    sx = StandardScaler().fit(x_train)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    xs = sx.transform(x_train)
    ys = sy.transform(y_train.reshape(-1, 1)).ravel()
    cut = max(2, int(len(xs) * 0.85))
    cut = min(cut, len(xs) - 1)
    xv, yv = xs[cut:], ys[cut:]
    xt, yt = xs[:cut], ys[:cut]
    net = DenseNetwork(xt.shape[1], hidden_size=48, seed=seed)
    net.fit(xt, yt, xv, yv, epochs=100, learning_rate=0.0015, batch_size=32, patience=15, seed=seed)
    model_val = sy.inverse_transform(net.predict(xv).reshape(-1, 1)).ravel()
    baseline_val = np.full(len(yv), float(np.mean(y_train[-6:])), dtype=np.float64)
    best_alpha = 0.0
    best_mae = float(np.mean(np.abs(baseline_val - yv)))
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * model_val + (1.0 - alpha) * baseline_val
        mae = float(np.mean(np.abs(blended - yv)))
        if mae + 1e-12 < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return net, sx, sy, best_alpha


def walk_forward_backtest(
    coin_id: str,
    interval: str,
    horizon: int,
    *,
    test_fraction: float = 0.20,
    refit_every: int | None = None,
    require_real_data: bool = True,
) -> dict:
    """Expanding-window, leakage-safe backtest.

    A model is refit periodically using only observations strictly before each
    prediction point. Validation data used to choose the ensemble weight is also
    contained entirely inside the training window.
    """
    coin_id = normalize_coin_id(coin_id)
    if interval not in HORIZONS or horizon not in HORIZONS[interval]:
        raise ValueError('Unsupported prediction interval or horizon.')

    candles, fallback = load_candles(coin_id, interval, force_refresh=False, allow_fallback=not require_real_data)
    if require_real_data and fallback:
        raise RuntimeError(f'No real CoinGecko data available for {coin_id}/{interval}. Refusing synthetic backtest.')

    x, y, _, _ = make_supervised(candles, horizon)
    n = len(x)
    test_start = max(10, int(n * (1.0 - test_fraction)))
    test_start = min(test_start, n - 1)

    predictions: list[float] = []
    actuals: list[float] = []
    baseline: list[float] = []

    net = sx = sy = None
    alpha = 0.0
    if refit_every is None:
        # Hourly data is large; daily data is small. These defaults are
        # deliberately explicit and configurable so the audit is repeatable.
        refit_every = 24 if interval == 'hourly' else 7
    refit_every = max(1, int(refit_every))

    started = time.perf_counter()
    for step, i in enumerate(range(test_start, n)):
        x_train = x[:i]
        y_train = y[:i]
        if net is None or step % refit_every == 0:
            net, sx, sy, alpha = _fit_model(x_train, y_train, seed=42 + i)
        pred_scaled = net.predict(sx.transform(x[i:i+1]))[0]
        model_pred = float(sy.inverse_transform(np.array([[pred_scaled]]))[0, 0])
        recent_mean = float(np.mean(y_train[-6:])) if len(y_train) else 0.0
        pred = alpha * model_pred + (1.0 - alpha) * recent_mean
        predictions.append(pred)
        actuals.append(float(y[i]))
        baseline.append(float(_baseline_predictions(y_train, 1)[0]))

    actual_arr = np.asarray(actuals, dtype=np.float64)
    pred_arr = np.asarray(predictions, dtype=np.float64)
    base_arr = np.asarray(baseline, dtype=np.float64)
    m = _metrics(actual_arr, pred_arr)
    b = _metrics(actual_arr, base_arr)
    metrics = BacktestMetrics(
        mae_pct=m[0], rmse_pct=m[1], mape_pct=m[2], directional_accuracy_pct=m[3],
        baseline_mae_pct=b[0], baseline_rmse_pct=b[1], baseline_mape_pct=b[2],
        baseline_directional_accuracy_pct=b[3],
        model_beats_baseline_mae=m[0] < b[0],
        model_beats_baseline_rmse=m[1] < b[1],
        model_beats_baseline_mape=m[2] < b[2],
        model_beats_baseline_directional_accuracy=m[3] > b[3],
    )
    return {
        'coin_id': coin_id,
        'coin_name': COIN_MAP[coin_id]['name'],
        'symbol': COIN_MAP[coin_id]['symbol'],
        'interval': interval,
        'horizon': horizon,
        'horizon_label': HORIZONS[interval][horizon],
        'samples_total': n,
        'test_samples': len(actual_arr),
        'refit_every': refit_every,
        'runtime_seconds': round(time.perf_counter() - started, 3),
        'data_source': 'Synthetic offline fallback' if fallback else 'CoinGecko market data',
        'metrics': metrics.to_dict(),
        'interpretation': _interpret(metrics),
    }


def _interpret(m: BacktestMetrics) -> str:
    wins = sum((m.model_beats_baseline_mae, m.model_beats_baseline_rmse, m.model_beats_baseline_mape, m.model_beats_baseline_directional_accuracy))
    if wins == 4:
        return 'The deployed ensemble beat the baseline on all four reported metrics in this backtest.'
    if wins == 0:
        return 'The deployed ensemble did not beat the baseline in this backtest. Treat the result as weak evidence, not a trading edge.'
    return f'The deployed ensemble beat the baseline on {wins}/4 metrics. Further testing across more periods is warranted.'

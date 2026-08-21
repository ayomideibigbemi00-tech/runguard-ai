import pytest
import numpy as np
from app.services.backtest import _metrics, _baseline_predictions


def test_metrics_are_percentage_errors():
    actual = np.array([0.01, -0.02, 0.03])
    pred = np.array([0.02, -0.01, 0.01])
    mae, rmse, mape, direction = _metrics(actual, pred)
    assert round(mae, 6) == round(np.mean(np.abs(pred-actual))*100, 6)
    assert rmse > 0
    assert 0 <= direction <= 100


def test_baseline_uses_only_training_tail():
    y = np.array([0.01, 0.03, -0.02, 0.04])
    out = _baseline_predictions(y, 3, lookback=2)
    assert np.allclose(out, np.mean(y[-2:]))

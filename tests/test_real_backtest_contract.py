from pathlib import Path

from app.services import backtest
from app.services.data import load_candles


def test_real_backtest_contract_rejects_synthetic(monkeypatch):
    import pandas as pd
    import numpy as np

    fake = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=200, freq='h', tz='UTC'),
        'open': np.arange(200, dtype=float) + 1,
        'high': np.arange(200, dtype=float) + 2,
        'low': np.arange(200, dtype=float) + 0.5,
        'close': np.arange(200, dtype=float) + 1,
        'volume': np.ones(200),
    })
    monkeypatch.setattr(backtest, 'load_candles', lambda *a, **k: (fake, True))
    try:
        backtest.walk_forward_backtest('bitcoin', 'hourly', 1, require_real_data=True, refit_every=24)
    except RuntimeError as exc:
        assert 'Refusing synthetic backtest' in str(exc)
    else:
        raise AssertionError('Synthetic data must never be accepted by the real-data audit.')

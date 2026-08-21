from datetime import datetime, timezone

import numpy as np
import pandas as pd


def test_prediction_anchors_money_value_to_live_price(monkeypatch, tmp_path):
    import app.services.predictor as predictor
    import app.services.predictions as storage

    monkeypatch.setattr(storage, 'PREDICTIONS_DIR', tmp_path)
    monkeypatch.setattr(storage, 'PREDICTIONS_PATH', tmp_path / 'predictions.jsonl')
    monkeypatch.setattr(storage, 'PREDICTIONS_CSV_PATH', tmp_path / 'prediction_history.csv')

    class IdentityScaler:
        def transform(self, value):
            return np.asarray(value, dtype=float)

        def inverse_transform(self, value):
            return np.asarray(value, dtype=float)

    class FakeNet:
        def predict(self, value):
            return np.array([0.10], dtype=float)

    candles = pd.DataFrame({
        'timestamp': pd.date_range('2026-08-21T08:00:00Z', periods=40, freq='h'),
        'close': np.linspace(45.0, 50.0, 40),
    })
    monkeypatch.setattr(predictor, 'load_candles', lambda *args, **kwargs: (candles, False))
    monkeypatch.setattr(predictor, '_train', lambda *args, **kwargs: (
        FakeNet(), IdentityScaler(), IdentityScaler(), candles, False, 0.1, 'neural_network', 1.0, 2.0, 1.0
    ))
    feature_rows = {name: np.ones(len(candles), dtype=float) for name in __import__('config').FEATURES}
    feature_rows['close'] = candles['close'].to_numpy()
    feature_rows['return_1'] = np.zeros(len(candles), dtype=float)
    monkeypatch.setattr(predictor, 'engineer_features', lambda _: pd.DataFrame(feature_rows))
    monkeypatch.setattr(predictor, 'fetch_current_prices', lambda: {
        'bitcoin': {
            'symbol': 'BTC', 'name': 'Bitcoin', 'price': 100.0,
            'last_updated_at': 1787300000,
            'observed_at_utc': '2026-08-21T09:00:00Z',
        }
    })

    result = predictor.predict('bitcoin', 'hourly', 1)

    assert result.current_price == 100.0
    assert result.current_price_observed_at_utc == '2026-08-21T09:00:00Z'
    assert abs(result.predicted_price - 110.0) < 1e-9
    record = storage.list_predictions(limit=1)[0]
    assert record['current_price'] == 100.0
    assert record['current_price_observed_at_utc'] == '2026-08-21T09:00:00Z'

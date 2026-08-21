from datetime import datetime, timezone, timedelta


def test_prediction_record_and_expiry_resolution(monkeypatch, tmp_path):
    import app.services.predictions as svc
    monkeypatch.setattr(svc, 'PREDICTIONS_DIR', tmp_path)
    monkeypatch.setattr(svc, 'PREDICTIONS_PATH', tmp_path / 'predictions.jsonl')
    monkeypatch.setattr(svc, 'PREDICTIONS_CSV_PATH', tmp_path / 'prediction_history.csv')

    created = datetime(2026, 8, 21, 9, 0, tzinfo=timezone.utc)
    record = svc.create_prediction(
        coin_id='ripple', interval='hourly', horizon=1,
        current_price=3.0, predicted_price=3.3, predicted_change=0.3,
        predicted_change_pct=10.0, predicted_direction='Bullish',
        source='CoinGecko market data', created_at=created,
    )
    assert record['status'] == 'pending'
    assert record['prediction_id']
    assert (tmp_path / 'prediction_history.csv').exists()

    import pandas as pd
    fake = pd.DataFrame({
        'timestamp': [pd.Timestamp('2026-08-21T09:00:00Z'), pd.Timestamp('2026-08-21T10:00:00Z')],
        'close': [3.0, 3.5],
    })
    monkeypatch.setattr(svc, '_fetch_market_chart', lambda coin_id, interval, days: fake)
    resolved = svc.resolve_due_predictions(now=created + timedelta(hours=1, minutes=1))
    assert len(resolved) == 1
    assert resolved[0]['actual_price'] == 3.5
    assert resolved[0]['direction_correct'] is True
    assert resolved[0]['status'] == 'resolved'

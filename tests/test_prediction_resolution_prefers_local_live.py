from datetime import datetime, timezone, timedelta

import pandas as pd


def test_resolution_prefers_local_live_observation(monkeypatch, tmp_path):
    import app.services.predictions as svc

    monkeypatch.setattr(svc, 'PREDICTIONS_DIR', tmp_path)
    monkeypatch.setattr(svc, 'PREDICTIONS_PATH', tmp_path / 'predictions.jsonl')
    monkeypatch.setattr(svc, 'PREDICTIONS_CSV_PATH', tmp_path / 'prediction_history.csv')

    created = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)
    svc.create_prediction(
        coin_id='bitcoin', interval='hourly', horizon=1,
        current_price=100.0, current_price_observed_at_utc='2026-08-21T10:00:00Z',
        predicted_price=105.0, predicted_change=5.0, predicted_change_pct=5.0,
        predicted_direction='Bullish', source='CoinGecko live + historical model', created_at=created,
    )

    import app.services.live_prices as live
    monkeypatch.setattr(live, 'LIVE_PRICES_DIR', tmp_path / 'live')
    monkeypatch.setattr(live, 'LIVE_PRICES_PATH', tmp_path / 'live' / 'live_prices.jsonl')
    monkeypatch.setattr(live, 'LIVE_PRICES_CSV_PATH', tmp_path / 'live' / 'live_price_history.csv')
    live.record_price_snapshot({'bitcoin': {'price': 108.0, 'last_updated_at': 1787300000}}, observed_at=created + timedelta(hours=1))

    def should_not_call(*args, **kwargs):
        raise AssertionError('Historical market chart should not be needed when a local live target observation exists.')

    monkeypatch.setattr(svc, '_fetch_market_chart', should_not_call)
    resolved = svc.resolve_due_predictions(now=created + timedelta(hours=1, seconds=10))

    assert len(resolved) == 1
    assert resolved[0]['actual_price'] == 108.0
    assert resolved[0]['actual_observation_time_utc'] == '2026-08-21T11:00:00Z'
    assert resolved[0]['direction_correct'] is True

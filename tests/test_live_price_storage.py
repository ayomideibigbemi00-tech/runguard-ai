from datetime import datetime, timezone


def test_live_price_snapshot_is_persisted(monkeypatch, tmp_path):
    import app.services.live_prices as svc

    monkeypatch.setattr(svc, 'LIVE_PRICES_DIR', tmp_path)
    monkeypatch.setattr(svc, 'LIVE_PRICES_PATH', tmp_path / 'live_prices.jsonl')
    monkeypatch.setattr(svc, 'LIVE_PRICES_CSV_PATH', tmp_path / 'live_price_history.csv')

    observed = datetime(2026, 8, 21, 11, 0, tzinfo=timezone.utc)
    rows = svc.record_price_snapshot({
        'bitcoin': {'price': 77891.66, 'last_updated_at': 1787300000},
        'ripple': {'price': 1.30, 'last_updated_at': 1787300000},
    }, observed_at=observed)

    assert len(rows) == 2
    assert (tmp_path / 'live_prices.jsonl').exists()
    assert (tmp_path / 'live_price_history.csv').exists()
    nearest = svc.find_nearest_observation('bitcoin', observed)
    assert nearest == (77891.66, '2026-08-21T11:00:00Z')


def test_fetch_current_prices_persists_the_batched_live_snapshot(monkeypatch, tmp_path):
    import io
    import json
    import app.services.data as data
    import app.services.live_prices as live

    monkeypatch.setattr(live, 'LIVE_PRICES_DIR', tmp_path)
    monkeypatch.setattr(live, 'LIVE_PRICES_PATH', tmp_path / 'live_prices.jsonl')
    monkeypatch.setattr(live, 'LIVE_PRICES_CSV_PATH', tmp_path / 'live_price_history.csv')
    monkeypatch.setattr(data, '_rate_limit_wait', lambda: None)

    payload = json.dumps({
        'bitcoin': {'usd': 77891.66, 'last_updated_at': 1787300000},
        'ripple': {'usd': 1.30, 'last_updated_at': 1787300000},
    }).encode('utf-8')

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(payload)
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(data, 'urlopen', lambda *args, **kwargs: FakeResponse())
    prices = data.fetch_current_prices()

    assert prices['bitcoin']['price'] == 77891.66
    assert prices['bitcoin']['observed_at_utc'].endswith('Z')
    assert live.latest_observation('bitcoin')['price'] == 77891.66

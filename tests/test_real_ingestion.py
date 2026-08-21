from app.services import data


def test_real_ingestion_never_uses_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(data, 'RETRY_QUEUE_PATH', tmp_path / 'retry_queue.json')
    monkeypatch.setattr(data, 'RATE_STATE_PATH', tmp_path / 'rate_state.json')
    monkeypatch.setattr(data, '_fetch_market_chart', lambda *a, **k: (_ for _ in ()).throw(data.RetryableCoinGeckoError('network down')))
    try:
        data.load_candles('bitcoin', 'hourly', force_refresh=True, allow_fallback=False)
    except data.RetryableCoinGeckoError:
        pass
    else:
        raise AssertionError('Strict ingestion must not silently synthesize data.')
    queue = data.get_retry_queue()
    assert queue and queue[0]['coin_id'] == 'bitcoin'

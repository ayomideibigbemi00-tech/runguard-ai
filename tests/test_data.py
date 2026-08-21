from app.services.data import _cache_path, _synthetic, normalize_coin_id, validate_coin


def test_each_coin_has_distinct_cache_key():
    assert _cache_path('bitcoin', 'hourly') != _cache_path('ethereum', 'hourly')


def test_synthetic_is_distinct_per_coin():
    btc = _synthetic('bitcoin', 'hourly', rows=100)
    eth = _synthetic('ethereum', 'hourly', rows=100)
    assert not btc['close'].equals(eth['close'])


def test_invalid_coin_rejected():
    try:
        validate_coin('made-up')
    except ValueError:
        return
    raise AssertionError('Invalid coin was accepted')


def test_xrp_uses_coin_gecko_ripple_id():
    assert normalize_coin_id('xrp') == 'ripple'
    assert normalize_coin_id('XRP') == 'ripple'
    assert _cache_path('xrp', 'hourly') == _cache_path('ripple', 'hourly')


def test_xrp_fetch_uses_ripple_api_path(monkeypatch):
    import json as _json
    from app.services import data

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return _json.dumps({
                'prices': [[1700000000000, 1.0], [1700003600000, 1.1], [1700007200000, 1.05]],
                'total_volumes': [[1700000000000, 100.0], [1700003600000, 110.0], [1700007200000, 105.0]],
            }).encode()

    seen = {}
    def fake_urlopen(request, timeout=None):
        seen['url'] = request.full_url
        return Response()

    monkeypatch.setattr(data, 'urlopen', fake_urlopen)
    monkeypatch.setattr(data, '_rate_limit_wait', lambda *a, **k: None)
    frame = data._fetch_market_chart('xrp', 'hourly', 1)
    assert '/coins/ripple/market_chart' in seen['url']
    assert not frame.empty
    assert list(frame.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

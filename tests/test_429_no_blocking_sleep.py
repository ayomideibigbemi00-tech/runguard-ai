from urllib.error import HTTPError
from io import BytesIO

from app.services import data


def test_429_is_deferred_to_queue_without_internal_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(data, '_rate_limit_wait', lambda: None)
    monkeypatch.setattr(data.time, 'sleep', lambda seconds: sleeps.append(seconds))

    def fake_urlopen(*args, **kwargs):
        raise HTTPError(
            url='https://example.test',
            code=429,
            msg='Too Many Requests',
            hdrs={},
            fp=BytesIO(b''),
        )

    monkeypatch.setattr(data, 'urlopen', fake_urlopen)

    try:
        data._fetch_market_chart('bitcoin', 'hourly', 100)
    except data.RateLimitError:
        pass
    else:
        raise AssertionError('Expected RateLimitError')

    assert sleeps == []

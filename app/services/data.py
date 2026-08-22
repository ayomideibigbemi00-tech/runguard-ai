from __future__ import annotations

import hashlib
from datetime import timezone
import json
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from config import (
    CACHE_DIR,
    CACHE_EXPIRY_HOURS,
    COIN_MAP,
    COINGECKO_API_KEY,
    COINGECKO_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    VS_CURRENCY,
)
from app.services.live_prices import record_price_snapshot

# In-memory cache for live prices
_LIVE_PRICE_CACHE = {}
_LIVE_PRICE_CACHE_LOCK = threading.Lock()
_LIVE_PRICE_CACHE_TIMESTAMP = 0
_LIVE_PRICE_CACHE_TTL = 60  # seconds

DEFAULT_REQUESTS_PER_MINUTE = 30
REQUESTS_PER_MINUTE = max(1, int(os.getenv('COINGECKO_REQUESTS_PER_MINUTE', str(DEFAULT_REQUESTS_PER_MINUTE))))
REQUEST_SPACING_SECONDS = 60.0 / REQUESTS_PER_MINUTE
MAX_RETRIES = max(0, int(os.getenv('COINGECKO_MAX_RETRIES', '6')))
RETRY_BASE_SECONDS = max(1.0, float(os.getenv('COINGECKO_RETRY_BASE_SECONDS', '5')))
RETRY_MAX_SECONDS = max(RETRY_BASE_SECONDS, float(os.getenv('COINGECKO_RETRY_MAX_SECONDS', '120')))
RETRY_QUEUE_PATH = CACHE_DIR / 'retry_queue.json'
RATE_STATE_PATH = CACHE_DIR / 'rate_state.json'

_RATE_LOCK = threading.Lock()
_REQUEST_TIMESTAMPS: deque[float] = deque()


class RateLimitError(RuntimeError):
    """Raised when CoinGecko continues to reject a request after safe retries."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RetryableCoinGeckoError(RuntimeError):
    """A transient request error that should be put back into a retry queue."""


class PermanentCoinGeckoError(RuntimeError):
    """A non-retryable CoinGecko error."""


COIN_ALIASES = {
    'xrp': 'ripple',
}


def normalize_coin_id(coin_id: str) -> str:
    normalized = str(coin_id).strip().lower()
    normalized = COIN_ALIASES.get(normalized, normalized)
    if normalized not in COIN_MAP:
        raise ValueError('Unsupported coin.')
    return normalized


def validate_coin(coin_id: str) -> None:
    normalize_coin_id(coin_id)


def _safe_key(value: str) -> str:
    return hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]


def _cache_path(coin_id: str, interval: str) -> Path:
    coin_id = normalize_coin_id(coin_id)
    return CACHE_DIR / f'{_safe_key(coin_id)}_{interval}.csv'


def _metadata_path(coin_id: str, interval: str) -> Path:
    coin_id = normalize_coin_id(coin_id)
    return CACHE_DIR / f'{_safe_key(coin_id)}_{interval}.json'


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def _load_rate_state() -> list[float]:
    values = _load_json(RATE_STATE_PATH, [])
    return [float(v) for v in values if isinstance(v, (int, float))]


def _save_rate_state(values: list[float]) -> None:
    cutoff = time.time() - 60.0
    _write_json(RATE_STATE_PATH, [ts for ts in values if ts > cutoff])


def _rate_limit_wait(extra_delay: float = 0.0) -> None:
    """Enforce a sliding-window RPM limit for every request, including retries."""
    with _RATE_LOCK:
        persisted = _load_rate_state()
        if not _REQUEST_TIMESTAMPS:
            _REQUEST_TIMESTAMPS.extend(persisted)
        else:
            existing = set(_REQUEST_TIMESTAMPS)
            _REQUEST_TIMESTAMPS.extend(ts for ts in persisted if ts not in existing)

        while True:
            now = time.time()
            while _REQUEST_TIMESTAMPS and _REQUEST_TIMESTAMPS[0] <= now - 60.0:
                _REQUEST_TIMESTAMPS.popleft()

            spacing_wait = 0.0
            if _REQUEST_TIMESTAMPS:
                spacing_wait = max(0.0, REQUEST_SPACING_SECONDS - (now - _REQUEST_TIMESTAMPS[-1]))

            window_wait = 0.0
            if len(_REQUEST_TIMESTAMPS) >= REQUESTS_PER_MINUTE:
                window_wait = max(0.0, (_REQUEST_TIMESTAMPS[0] + 60.0) - now)

            wait_for = max(spacing_wait, window_wait, extra_delay)
            if wait_for <= 0:
                break
            time.sleep(wait_for)

        timestamp = time.time()
        _REQUEST_TIMESTAMPS.append(timestamp)
        _save_rate_state(list(_REQUEST_TIMESTAMPS))


def _retry_after_seconds(exc: HTTPError) -> float | None:
    value = exc.headers.get('Retry-After') if exc.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _queue_key(coin_id: str, interval: str) -> str:
    return f'{coin_id}|{interval}'


def _load_retry_queue() -> list[dict]:
    queue = _load_json(RETRY_QUEUE_PATH, [])
    return queue if isinstance(queue, list) else []


def _save_retry_queue(queue: list[dict]) -> None:
    dedup = {}
    for item in queue:
        dedup[_queue_key(item['coin_id'], item['interval'])] = item
    _write_json(RETRY_QUEUE_PATH, list(dedup.values()))


def enqueue_retry(
    coin_id: str,
    interval: str,
    *,
    reason: str,
    attempt: int,
    not_before: float | None = None,
) -> dict:
    item = {
        'coin_id': coin_id,
        'interval': interval,
        'attempt': int(attempt),
        'reason': reason,
        'not_before': float(not_before or time.time()),
        'updated_at': time.time(),
    }
    queue = _load_retry_queue()
    queue = [q for q in queue if _queue_key(q['coin_id'], q['interval']) != _queue_key(coin_id, interval)]
    queue.append(item)
    _save_retry_queue(queue)
    return item


def remove_retry(coin_id: str, interval: str) -> None:
    queue = _load_retry_queue()
    key = _queue_key(coin_id, interval)
    _save_retry_queue([q for q in queue if _queue_key(q['coin_id'], q['interval']) != key])


def get_retry_queue() -> list[dict]:
    return sorted(_load_retry_queue(), key=lambda x: (x.get('not_before', 0), x.get('coin_id', '')))


def retry_queue_status() -> dict:
    queue = get_retry_queue()
    now = time.time()
    ready = [q for q in queue if float(q.get('not_before', 0)) <= now]
    return {'pending': len(queue), 'ready': len(ready), 'items': queue}


def _fetch_market_chart(coin_id: str, interval: str, days: int) -> pd.DataFrame:
    coin_id = normalize_coin_id(coin_id)
    api_interval = 'hourly' if interval == 'hourly' else 'daily'
    params = {'vs_currency': VS_CURRENCY, 'days': str(days), 'interval': api_interval}
    url = f'{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart?{urlencode(params)}'
    headers = {'User-Agent': 'Runguard-AI/4.1'}
    if COINGECKO_API_KEY:
        headers['x-cg-demo-api-key'] = COINGECKO_API_KEY

    _rate_limit_wait()
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        retry_after = _retry_after_seconds(exc)
        retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
        if not retryable:
            raise PermanentCoinGeckoError(f'CoinGecko HTTP {exc.code} for {coin_id}/{interval}.') from exc
        if exc.code == 429:
            raise RateLimitError(
                f'CoinGecko HTTP 429 for {coin_id}/{interval}. '
                'The job has been persisted for later retry.',
                retry_after=retry_after,
            ) from exc
        raise RetryableCoinGeckoError(
            f'CoinGecko transient HTTP {exc.code} for {coin_id}/{interval}. '
            'The job has been persisted for later retry.'
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RetryableCoinGeckoError(
            f'CoinGecko network request failed for {coin_id}/{interval}: {exc}'
        ) from exc

    prices = pd.DataFrame(data.get('prices', []), columns=['timestamp', 'price'])
    volumes = pd.DataFrame(data.get('total_volumes', []), columns=['timestamp', 'volume'])
    if prices.empty:
        raise PermanentCoinGeckoError(f'CoinGecko returned no price data for {coin_id}.')

    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms', utc=True)
    prices = prices.sort_values('timestamp').drop_duplicates('timestamp')
    if not volumes.empty:
        volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms', utc=True)
        volumes = volumes.sort_values('timestamp').drop_duplicates('timestamp')
    df = prices.merge(volumes, on='timestamp', how='left').set_index('timestamp')

    rule = '1h' if interval == 'hourly' else '1D'
    candles = df['price'].resample(rule).ohlc()
    vol = df['volume'].resample(rule).sum(min_count=1).rename('volume')
    candles = candles.join(vol).ffill().dropna().reset_index()
    candles.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return candles


def _write_cache(coin_id: str, interval: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(coin_id, interval)
    tmp = path.with_suffix('.csv.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    _write_json(_metadata_path(coin_id, interval), {
        'coin_id': coin_id,
        'interval': interval,
        'source': 'CoinGecko market data',
        'cached_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'last_timestamp': str(df['timestamp'].iloc[-1]),
        'rows': int(len(df)),
    })


def _read_cache(coin_id: str, interval: str) -> pd.DataFrame:
    return pd.read_csv(_cache_path(coin_id, interval), parse_dates=['timestamp'])


def cache_info(coin_id: str, interval: str) -> dict:
    coin_id = normalize_coin_id(coin_id)
    path = _cache_path(coin_id, interval)
    meta = _load_json(_metadata_path(coin_id, interval), {})
    if not path.exists():
        return {'exists': False, 'fresh': False, **meta}
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return {
        'exists': True,
        'fresh': age_hours < CACHE_EXPIRY_HOURS,
        'age_hours': round(age_hours, 3),
        **meta,
    }


def _synthetic(coin_id: str, interval: str, rows: int = 1500) -> pd.DataFrame:
    seed = int(hashlib.sha256(f'{coin_id}:{interval}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(hours=1 if interval == 'hourly' else 24)
    end = pd.Timestamp.now(tz='UTC').floor('h')
    ts = pd.date_range(end=end, periods=rows, freq=step)
    base_prices = np.array([62000.0, 2800.0, 1.0, 600.0, 150.0])
    base = base_prices[seed % len(base_prices)]
    drift = 0.00003 if interval == 'hourly' else 0.0008
    noise = rng.normal(drift, 0.012 if interval == 'hourly' else 0.025, size=rows)
    close = base * np.exp(np.cumsum(noise))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.004, rows)) * close
    high = np.maximum(open_, close) + spread
    low = np.maximum(0.0, np.minimum(open_, close) - spread)
    volume = rng.lognormal(mean=15.5 if interval == 'hourly' else 17.5, sigma=0.35, size=rows)
    return pd.DataFrame({'timestamp': ts, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})


def load_candles(
    coin_id: str,
    interval: str,
    force_refresh: bool = False,
    allow_fallback: bool = True,
) -> tuple[pd.DataFrame, bool]:
    coin_id = normalize_coin_id(coin_id)
    if interval not in {'hourly', 'daily'}:
        raise ValueError('interval must be hourly or daily')

    path = _cache_path(coin_id, interval)
    info = cache_info(coin_id, interval)
    if info.get('fresh') and not force_refresh:
        return _read_cache(coin_id, interval), False

    days = 100 if interval == 'hourly' else 365
    try:
        df = _fetch_market_chart(coin_id, interval, days)
        _write_cache(coin_id, interval, df)
        remove_retry(coin_id, interval)
        return df, False
    except (RateLimitError, RetryableCoinGeckoError) as exc:
        not_before = time.time() + (exc.retry_after or RETRY_MAX_SECONDS if isinstance(exc, RateLimitError) else RETRY_BASE_SECONDS)
        attempt = int(next((q.get('attempt', 0) for q in _load_retry_queue() if _queue_key(q['coin_id'], q['interval']) == _queue_key(coin_id, interval)), 0)) + 1
        enqueue_retry(coin_id, interval, reason=str(exc), attempt=attempt, not_before=not_before)
        if path.exists() and allow_fallback:
            return _read_cache(coin_id, interval), False
        raise
    except PermanentCoinGeckoError:
        if path.exists() and allow_fallback:
            return _read_cache(coin_id, interval), False
        raise
    except Exception:
        if path.exists() and allow_fallback:
            return _read_cache(coin_id, interval), False
        if allow_fallback:
            return _synthetic(coin_id, interval), True
        raise


def ingest_jobs(
    jobs: list[tuple[str, str]],
    *,
    force_refresh: bool = False,
    max_passes: int = 10,
) -> dict:
    pending = {(coin_id, interval) for coin_id, interval in jobs}
    pending.update((_q['coin_id'], _q['interval']) for _q in get_retry_queue())
    completed = []
    failures = []
    passed = 0

    while pending and passed < max_passes:
        passed += 1
        progress = False
        for coin_id, interval in list(sorted(pending)):
            q = next((item for item in get_retry_queue() if _queue_key(item['coin_id'], item['interval']) == _queue_key(coin_id, interval)), None)
            if q and time.time() < float(q.get('not_before', 0)):
                continue
            try:
                df, fallback = load_candles(coin_id, interval, force_refresh=force_refresh, allow_fallback=False)
                if fallback:
                    raise RuntimeError('Synthetic fallback returned during ingest.')
                completed.append({'coin_id': coin_id, 'interval': interval, 'rows': len(df)})
                remove_retry(coin_id, interval)
                pending.remove((coin_id, interval))
                progress = True
            except Exception as exc:
                failures.append({'coin_id': coin_id, 'interval': interval, 'error': str(exc)})

        if pending and not progress:
            queue = get_retry_queue()
            eligible_times = [float(q['not_before']) for q in queue if (q['coin_id'], q['interval']) in pending]
            if eligible_times:
                wait = max(0.0, min(eligible_times) - time.time())
                if wait > 0:
                    time.sleep(wait)
            else:
                break

    return {
        'requested': len(jobs),
        'completed': completed,
        'pending_retry_queue': get_retry_queue(),
        'failures_observed': failures,
        'passes': passed,
    }


def fetch_current_prices() -> dict[str, dict]:
    """Fetch current prices with 24h change and in-memory caching."""
    global _LIVE_PRICE_CACHE, _LIVE_PRICE_CACHE_TIMESTAMP

    now = time.time()
    with _LIVE_PRICE_CACHE_LOCK:
        if _LIVE_PRICE_CACHE and (now - _LIVE_PRICE_CACHE_TIMESTAMP) < _LIVE_PRICE_CACHE_TTL:
            return _LIVE_PRICE_CACHE

    from config import COINS

    ids = [normalize_coin_id(coin['id']) for coin in COINS]
    params = {
        'ids': ','.join(ids),
        'vs_currencies': VS_CURRENCY,
        'include_last_updated_at': 'true',
        'include_24hr_change': 'true',
    }
    url = f'{COINGECKO_BASE_URL}/simple/price?{urlencode(params)}'
    headers = {'User-Agent': 'Runguard-AI/4.1'}
    if COINGECKO_API_KEY:
        headers['x-cg-demo-api-key'] = COINGECKO_API_KEY

    _rate_limit_wait()
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        retry_after = _retry_after_seconds(exc)
        retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}
        if exc.code == 429:
            raise RateLimitError('CoinGecko current-price request returned HTTP 429.', retry_after=retry_after) from exc
        if retryable:
            raise RetryableCoinGeckoError(f'CoinGecko current-price request returned HTTP {exc.code}.') from exc
        raise PermanentCoinGeckoError(f'CoinGecko current-price request returned HTTP {exc.code}.') from exc
    except (URLError, TimeoutError) as exc:
        raise RetryableCoinGeckoError(f'CoinGecko current-price request failed: {exc}') from exc

    observed_at = pd.Timestamp.now(tz='UTC').to_pydatetime()
    output = {}
    by_id = {coin['id']: coin for coin in COINS}
    for coin_id, values in data.items():
        if coin_id not in by_id or not isinstance(values, dict) or VS_CURRENCY not in values:
            continue
        output[coin_id] = {
            'symbol': by_id[coin_id]['symbol'],
            'name': by_id[coin_id]['name'],
            'price': float(values[VS_CURRENCY]),
            'last_updated_at': values.get('last_updated_at'),
            'observed_at_utc': observed_at.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'price_change_percentage_24h': values.get('usd_24h_change', 0.0),
        }

    with _LIVE_PRICE_CACHE_LOCK:
        _LIVE_PRICE_CACHE = output
        _LIVE_PRICE_CACHE_TIMESTAMP = time.time()

    try:
        record_price_snapshot(output, observed_at=observed_at)
    except OSError:
        pass
    return output
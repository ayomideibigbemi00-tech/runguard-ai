from __future__ import annotations

import csv
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import CACHE_DIR, COIN_MAP

LIVE_PRICES_DIR = CACHE_DIR / 'live_prices'
LIVE_PRICES_PATH = LIVE_PRICES_DIR / 'live_prices.jsonl'
LIVE_PRICES_CSV_PATH = LIVE_PRICES_DIR / 'live_price_history.csv'
LIVE_PRICE_FIELDS = [
    'observation_id',
    'coin_id',
    'symbol',
    'coin_name',
    'price',
    'observed_at_utc',
    'provider_last_updated_at',
    'source',
]
_LIVE_PRICE_LOCK = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _write_csv_header_if_needed(handle) -> csv.DictWriter:
    writer = csv.DictWriter(handle, fieldnames=LIVE_PRICE_FIELDS, extrasaction='ignore')
    if handle.tell() == 0:
        writer.writeheader()
    return writer


def record_price_snapshot(
    prices: dict[str, dict[str, Any]],
    *,
    observed_at: datetime | None = None,
    source: str = 'CoinGecko simple/price',
) -> list[dict[str, Any]]:
    """Append one local observation for every price returned in a live snapshot."""
    if not prices:
        return []
    observed_at = observed_at or _utc_now()
    observed_at_utc = _iso(observed_at)
    rows: list[dict[str, Any]] = []
    for coin_id, value in prices.items():
        if coin_id not in COIN_MAP or not isinstance(value, dict):
            continue
        price = value.get('price')
        if price is None:
            continue
        rows.append({
            'observation_id': uuid.uuid4().hex,
            'coin_id': coin_id,
            'symbol': COIN_MAP[coin_id]['symbol'],
            'coin_name': COIN_MAP[coin_id]['name'],
            'price': float(price),
            'observed_at_utc': observed_at_utc,
            'provider_last_updated_at': value.get('last_updated_at'),
            'source': source,
        })
    if not rows:
        return []

    with _LIVE_PRICE_LOCK:
        LIVE_PRICES_DIR.mkdir(parents=True, exist_ok=True)
        with LIVE_PRICES_PATH.open('a', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(',', ':'), sort_keys=True) + '\n')
        with LIVE_PRICES_CSV_PATH.open('a', newline='', encoding='utf-8') as handle:
            writer = _write_csv_header_if_needed(handle)
            writer.writerows(rows)
    return rows


def _read_records() -> list[dict[str, Any]]:
    if not LIVE_PRICES_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    with LIVE_PRICES_PATH.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def find_nearest_observation(
    coin_id: str,
    target_time: datetime,
    *,
    max_distance_seconds: float | None = None,
) -> tuple[float, str] | None:
    """Return the nearest locally recorded live observation to target_time."""
    target = target_time.astimezone(timezone.utc)
    with _LIVE_PRICE_LOCK:
        records = _read_records()
    candidates = [record for record in records if record.get('coin_id') == coin_id]
    if not candidates:
        return None

    best = None
    best_distance = None
    for record in candidates:
        try:
            observed = datetime.fromisoformat(str(record['observed_at_utc']).replace('Z', '+00:00'))
            distance = abs((observed - target).total_seconds())
            if best_distance is None or distance < best_distance:
                best = record
                best_distance = distance
        except (KeyError, TypeError, ValueError):
            continue
    if best is None or best_distance is None:
        return None
    if max_distance_seconds is not None and best_distance > max_distance_seconds:
        return None
    return float(best['price']), str(best['observed_at_utc'])


def latest_observation(coin_id: str) -> dict[str, Any] | None:
    with _LIVE_PRICE_LOCK:
        records = _read_records()
    candidates = [record for record in records if record.get('coin_id') == coin_id]
    if not candidates:
        return None
    return max(candidates, key=lambda record: str(record.get('observed_at_utc', '')))

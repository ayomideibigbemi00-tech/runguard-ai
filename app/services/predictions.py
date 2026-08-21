from __future__ import annotations

import csv
import json
import math
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import CACHE_DIR, COIN_MAP, HORIZONS
from app.services.data import _fetch_market_chart, fetch_current_prices, normalize_coin_id
from app.services.live_prices import find_nearest_observation

PREDICTIONS_DIR = CACHE_DIR / 'predictions'
PREDICTIONS_PATH = PREDICTIONS_DIR / 'predictions.jsonl'
PREDICTIONS_CSV_PATH = PREDICTIONS_DIR / 'prediction_history.csv'
_PREDICTION_LOCK = threading.Lock()

FIELDS = [
    'prediction_id', 'coin_id', 'symbol', 'coin_name', 'interval', 'horizon',
    'horizon_label', 'created_at_utc', 'target_time_utc', 'current_price', 'current_price_observed_at_utc',
    'predicted_price', 'predicted_change', 'predicted_change_pct',
    'predicted_direction', 'actual_price', 'actual_change', 'actual_change_pct',
    'actual_direction', 'absolute_error', 'absolute_error_pct',
    'direction_correct', 'status', 'source', 'resolved_at_utc',
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _read_records() -> list[dict[str, Any]]:
    if not PREDICTIONS_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    with PREDICTIONS_PATH.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError:
                continue
    return records


def _write_records(records: list[dict[str, Any]]) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PREDICTIONS_PATH.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(',', ':'), sort_keys=True) + '\n')
    tmp.replace(PREDICTIONS_PATH)

    tmp_csv = PREDICTIONS_CSV_PATH.with_suffix('.csv.tmp')
    with tmp_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    tmp_csv.replace(PREDICTIONS_CSV_PATH)


def _direction(change: float) -> str:
    if change > 0:
        return 'Bullish'
    if change < 0:
        return 'Bearish'
    return 'Neutral'


def create_prediction(
    *,
    coin_id: str,
    interval: str,
    horizon: int,
    current_price: float,
    current_price_observed_at_utc: str | None = None,
    predicted_price: float,
    predicted_change: float,
    predicted_change_pct: float,
    predicted_direction: str,
    source: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    coin_id = normalize_coin_id(coin_id)
    if horizon not in HORIZONS.get(interval, {}):
        raise ValueError('Unsupported prediction horizon.')
    created_at = created_at or _utc_now()
    target = created_at + timedelta(hours=horizon if interval == 'hourly' else 24 * horizon)
    record = {
        'prediction_id': uuid.uuid4().hex,
        'coin_id': coin_id,
        'symbol': COIN_MAP[coin_id]['symbol'],
        'coin_name': COIN_MAP[coin_id]['name'],
        'interval': interval,
        'horizon': int(horizon),
        'horizon_label': HORIZONS[interval][horizon],
        'created_at_utc': _iso(created_at),
        'target_time_utc': _iso(target),
        'current_price': float(current_price),
        'current_price_observed_at_utc': current_price_observed_at_utc or _iso(created_at),
        'predicted_price': float(predicted_price),
        'predicted_change': float(predicted_change),
        'predicted_change_pct': float(predicted_change_pct),
        'predicted_direction': str(predicted_direction),
        'actual_price': None,
        'actual_change': None,
        'actual_change_pct': None,
        'actual_direction': None,
        'absolute_error': None,
        'absolute_error_pct': None,
        'direction_correct': None,
        'status': 'pending',
        'source': source,
        'resolved_at_utc': None,
    }
    with _PREDICTION_LOCK:
        records = _read_records()
        records.append(record)
        _write_records(records)
    return record


def _price_near_target(df: pd.DataFrame, target_time: datetime) -> tuple[float, str]:
    if df.empty:
        raise ValueError('No market prices returned for target resolution.')
    series = df.copy()
    series['timestamp'] = pd.to_datetime(series['timestamp'], utc=True)
    target = pd.Timestamp(target_time.astimezone(timezone.utc))
    series['distance'] = (series['timestamp'] - target).abs()
    row = series.sort_values('distance').iloc[0]
    return float(row['close']), _iso(row['timestamp'].to_pydatetime())


def resolve_due_predictions(now: datetime | None = None) -> list[dict[str, Any]]:
    """Resolve expired predictions from locally observed live prices first.

    When a target is recent, a fresh live-price request is made so a one-hour
    prediction can be resolved against the market that just reached its target.
    If the target was missed by a longer period, the local observation ledger is
    preferred; the historical CoinGecko candle series is only a recovery fallback.
    """
    now = now or _utc_now()
    with _PREDICTION_LOCK:
        records = _read_records()
        due = [r for r in records if r.get('status') == 'pending' and r.get('target_time_utc') and r['target_time_utc'] <= _iso(now)]
        if not due:
            return []

        fresh_prices: dict[str, dict[str, Any]] = {}
        # Only fetch the live catalogue when at least one target is close enough
        # that the current price is a defensible target observation.
        near_due = False
        for record in due:
            target_dt = datetime.fromisoformat(record['target_time_utc'].replace('Z', '+00:00'))
            if abs((now - target_dt).total_seconds()) <= 300:
                near_due = True
                break
        if near_due:
            try:
                fresh_prices = fetch_current_prices()
            except Exception:
                fresh_prices = {}

        historical_cache: dict[tuple[str, str], pd.DataFrame] = {}
        resolved: list[dict[str, Any]] = []
        for record in due:
            coin_id = record['coin_id']
            interval = record['interval']
            target_dt = datetime.fromisoformat(record['target_time_utc'].replace('Z', '+00:00'))
            actual: tuple[float, str] | None = None

            # Prefer a genuinely live observation taken within five minutes of
            # the target. The browser's periodic price refreshes populate this ledger.
            if coin_id in fresh_prices:
                live = fresh_prices[coin_id]
                observed_at = str(live.get('observed_at_utc', ''))
                try:
                    observed_dt = datetime.fromisoformat(observed_at.replace('Z', '+00:00'))
                    if abs((observed_dt - target_dt).total_seconds()) <= 300:
                        actual = (float(live['price']), observed_at)
                except (TypeError, ValueError, KeyError):
                    actual = None

            if actual is None:
                actual = find_nearest_observation(coin_id, target_dt, max_distance_seconds=900)

            if actual is None:
                key = (coin_id, interval)
                if key not in historical_cache:
                    days = 100 if interval == 'hourly' else 365
                    try:
                        historical_cache[key] = _fetch_market_chart(coin_id, interval, days)
                    except Exception as exc:
                        record['last_resolution_error'] = str(exc)
                        continue
                try:
                    actual = _price_near_target(historical_cache[key], target_dt)
                except Exception as exc:
                    record['last_resolution_error'] = str(exc)
                    continue

            try:
                actual_price, observation_time = actual
                current_price = float(record['current_price'])
                predicted_change = float(record['predicted_price']) - current_price
                actual_change = actual_price - current_price
                actual_change_pct = (actual_change / current_price * 100.0) if current_price else 0.0
                abs_error = abs(float(record['predicted_price']) - actual_price)
                abs_error_pct = (abs_error / actual_price * 100.0) if actual_price else 0.0
                actual_direction = _direction(actual_change)
                predicted_direction = str(record.get('predicted_direction', 'Neutral'))
                direction_correct = (
                    (predicted_direction == 'Bullish' and actual_change > 0)
                    or (predicted_direction == 'Bearish' and actual_change < 0)
                    or (predicted_direction == 'Neutral' and math.isclose(actual_change, 0.0, abs_tol=1e-12))
                )
                record.update({
                    'actual_price': actual_price,
                    'actual_change': actual_change,
                    'actual_change_pct': actual_change_pct,
                    'actual_direction': actual_direction,
                    'absolute_error': abs_error,
                    'absolute_error_pct': abs_error_pct,
                    'direction_correct': bool(direction_correct),
                    'status': 'resolved',
                    'resolved_at_utc': _iso(now),
                    'actual_observation_time_utc': observation_time,
                    'last_resolution_error': None,
                })
                resolved.append(record)
            except Exception as exc:
                record['last_resolution_error'] = str(exc)

        _write_records(records)
        return resolved

def list_predictions(*, coin_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with _PREDICTION_LOCK:
        records = _read_records()
    if coin_id:
        coin_id = normalize_coin_id(coin_id)
        records = [r for r in records if r.get('coin_id') == coin_id]
    if status:
        records = [r for r in records if r.get('status') == status]
    records.sort(key=lambda r: r.get('created_at_utc', ''), reverse=True)
    return records[:max(1, min(int(limit), 500))]


def prediction_summary() -> dict[str, Any]:
    resolve_due_predictions()
    with _PREDICTION_LOCK:
        records = _read_records()
    resolved = [r for r in records if r.get('status') == 'resolved']
    correct = [r for r in resolved if r.get('direction_correct') is True]
    return {
        'total': len(records),
        'pending': sum(r.get('status') == 'pending' for r in records),
        'resolved': len(resolved),
        'correct': len(correct),
        'wrong': sum(r.get('direction_correct') is False for r in resolved),
        'directional_accuracy_pct': (len(correct) / len(resolved) * 100.0) if resolved else None,
        'history_path': str(PREDICTIONS_CSV_PATH),
    }

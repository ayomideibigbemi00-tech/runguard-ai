from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import CACHE_DIR
from app.services.data import fetch_current_prices, normalize_coin_id

PREDICTIONS_DIR = CACHE_DIR / 'predictions'
PREDICTIONS_FILE = PREDICTIONS_DIR / 'predictions.jsonl'


def _ensure_dir():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _read_records() -> list[dict]:
    if not PREDICTIONS_FILE.exists():
        return []
    records = []
    with PREDICTIONS_FILE.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _write_records(records: list[dict]) -> None:
    _ensure_dir()
    with PREDICTIONS_FILE.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def save_prediction(coin_id, interval, horizon, predicted_price, current_price, horizon_label, strategy, validation_mae_pct, baseline_validation_mae_pct, target_time_utc):
    record = {
        'prediction_id': str(uuid.uuid4()),
        'coin_id': coin_id,
        'interval': interval,
        'horizon': horizon,
        'predicted_price': float(predicted_price),
        'current_price': float(current_price),  # Live price at prediction
        'horizon_label': horizon_label,
        'strategy': strategy,
        'validation_mae_pct': validation_mae_pct,
        'baseline_validation_mae_pct': baseline_validation_mae_pct,
        'target_time_utc': target_time_utc,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'pending',
        'actual_price': None,  # Live price after timeframe
        'absolute_error': None,
        'percentage_accuracy': None,
        'direction_correct': None,
    }
    records = _read_records()
    records.append(record)
    _write_records(records)
    return record


def list_predictions(limit: int = 100, status: Optional[str] = None, coin_id: Optional[str] = None) -> list[dict]:
    records = _read_records()
    if coin_id:
        records = [r for r in records if r.get('coin_id') == coin_id]
    if status:
        records = [r for r in records if r.get('status') == status]
    records = sorted(records, key=lambda x: x.get('created_at_utc', ''), reverse=True)
    return records[:limit]


def resolve_due_predictions() -> list[dict]:
    """
    Resolves all pending predictions whose target time has passed.
    Fetches current live price for the coin to determine the actual price.
    Updates accuracy and direction.
    """
    records = _read_records()
    resolved_count = 0
    now = datetime.now(timezone.utc)

    for record in records:
        if record.get('status') != 'pending':
            continue
        
        target_time = record.get('target_time_utc')
        if not target_time:
            continue
        
        try:
            target_dt = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
        except ValueError:
            continue

        if now < target_dt:
            continue

        # Fetch actual live price after timeframe
        try:
            prices = fetch_current_prices()
            coin_id = record.get('coin_id')
            actual_price = prices.get(coin_id, {}).get('price')
        except Exception:
            actual_price = None

        if actual_price:
            predicted = record.get('predicted_price')
            # Calculate percentage accuracy: 100% - (|Predicted - Actual| / Actual * 100)
            abs_error = abs(predicted - actual_price)
            accuracy = max(0.0, 100.0 - (abs_error / actual_price * 100.0))
            
            record['status'] = 'resolved'
            record['actual_price'] = actual_price  # Live price after timeframe
            record['absolute_error'] = abs_error
            record['percentage_accuracy'] = round(accuracy, 2)
            record['direction_correct'] = (predicted > actual_price) == (record.get('current_price', 0) < actual_price)
            resolved_count += 1

    if resolved_count > 0:
        _write_records(records)
    
    return [r for r in records if r.get('status') == 'resolved']


def prediction_summary() -> dict:
    records = _read_records()
    resolved = [r for r in records if r.get('status') == 'resolved']
    pending = [r for r in records if r.get('status') == 'pending']
    
    if not resolved:
        return {'total': len(records), 'resolved': 0, 'pending': len(pending), 'avg_accuracy': None, 'correct_count': 0}
    
    accuracies = [r.get('percentage_accuracy', 0) for r in resolved if r.get('percentage_accuracy') is not None]
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
    correct = sum(1 for r in resolved if r.get('direction_correct'))
    
    return {
        'total': len(records),
        'resolved': len(resolved),
        'pending': len(pending),
        'avg_accuracy': round(avg_acc, 2),
        'correct_count': correct,
        'total_resolved': len(resolved)
    }
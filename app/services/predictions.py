from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import CACHE_DIR
from app.db import get_db

PREDICTIONS_DIR = CACHE_DIR / 'predictions'
PREDICTIONS_FILE = PREDICTIONS_DIR / 'predictions.jsonl'


def _ensure_dir():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_prediction(user_id, coin_id, interval, horizon, predicted_price, current_price, horizon_label, strategy, validation_mae_pct, baseline_validation_mae_pct, target_time_utc):
    """Save prediction to SQLite database."""
    conn = get_db()
    cur = conn.execute('''
        INSERT INTO predictions (
            user_id, coin_id, interval, horizon, predicted_price, current_price,
            horizon_label, strategy, validation_mae_pct, baseline_validation_mae_pct,
            target_time_utc, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
    ''', (
        user_id, coin_id, interval, horizon, predicted_price, current_price,
        horizon_label, strategy, validation_mae_pct, baseline_validation_mae_pct,
        target_time_utc
    ))
    conn.commit()
    prediction_id = cur.lastrowid
    conn.close()
    return {'prediction_id': prediction_id}


def list_predictions(user_id: int, limit: int = 100, status: Optional[str] = None, coin_id: Optional[str] = None) -> list[dict]:
    conn = get_db()
    query = "SELECT * FROM predictions WHERE user_id = ?"
    params = [user_id]
    if coin_id:
        query += " AND coin_id = ?"
        params.append(coin_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at_utc DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    records = []
    for row in rows:
        record = dict(row)
        # Convert id to prediction_id for compatibility
        record['prediction_id'] = record.pop('id')
        records.append(record)
    return records


def resolve_due_predictions(user_id: int | None = None) -> list[dict]:
    """Resolve pending predictions whose target time has passed.
    
    Uses live price first, but falls back to the last known historical price
    if the live API is unavailable (e.g. rate limited).
    """
    conn = get_db()
    now = datetime.now(timezone.utc)
    
    # Get pending predictions
    if user_id:
        rows = conn.execute("SELECT * FROM predictions WHERE user_id = ? AND status = 'pending'", (user_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM predictions WHERE status = 'pending'").fetchall()
    
    resolved = []
    for row in rows:
        target_time = row['target_time_utc']
        if not target_time:
            continue
        
        try:
            target_dt = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
        except ValueError:
            continue

        if now < target_dt:
            continue  # Still pending

        actual_price = None
        # 1. Try live price (which can fail if rate limited)
        try:
            from app.services.data import fetch_current_prices
            prices = fetch_current_prices()
            actual_price = prices.get(row['coin_id'], {}).get('price')
        except Exception:
            actual_price = None
        
        # 2. Fallback to historical data if live price failed
        if actual_price is None:
            try:
                from app.services.data import load_candles
                df, fallback = load_candles(row['coin_id'], row['interval'], allow_fallback=True)
                if df is not None and not df.empty:
                    actual_price = float(df['close'].iloc[-1])
            except Exception:
                actual_price = None

        if actual_price:
            predicted = row['predicted_price']
            abs_error = abs(predicted - actual_price)
            accuracy = max(0.0, 100.0 - (abs_error / actual_price * 100.0))
            direction_correct = (predicted > row['current_price']) == (actual_price > row['current_price'])
            
            conn.execute('''
                UPDATE predictions
                SET status = 'resolved', actual_price = ?, absolute_error = ?, percentage_accuracy = ?, direction_correct = ?
                WHERE id = ?
            ''', (actual_price, abs_error, round(accuracy, 2), int(direction_correct), row['id']))
            resolved.append(dict(row))
    conn.commit()
    conn.close()
    return resolved


def prediction_summary(user_id: int) -> dict:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,)).fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND status = 'resolved'", (user_id,)).fetchone()[0]
    pending = total - resolved
    accuracies = conn.execute("SELECT percentage_accuracy FROM predictions WHERE user_id = ? AND percentage_accuracy IS NOT NULL", (user_id,)).fetchall()
    avg_acc = round(sum([a[0] for a in accuracies]) / len(accuracies), 2) if accuracies else None
    correct_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND direction_correct = 1", (user_id,)).fetchone()[0]
    conn.close()
    return {
        'total': total,
        'resolved': resolved,
        'pending': pending,
        'avg_accuracy': avg_acc,
        'correct_count': correct_count,
        'total_resolved': resolved
    }
import json
import time
from pathlib import Path

import pandas as pd

from app.services import data


def test_retry_queue_persists_and_can_be_removed(tmp_path, monkeypatch):
    queue_path = tmp_path / 'retry_queue.json'
    state_path = tmp_path / 'rate_state.json'
    monkeypatch.setattr(data, 'RETRY_QUEUE_PATH', queue_path)
    monkeypatch.setattr(data, 'RATE_STATE_PATH', state_path)
    monkeypatch.setattr(data, 'CACHE_DIR', tmp_path)

    item = data.enqueue_retry('bitcoin', 'hourly', reason='429', attempt=1, not_before=time.time())
    assert item['coin_id'] == 'bitcoin'
    assert data.get_retry_queue()[0]['interval'] == 'hourly'

    data.remove_retry('bitcoin', 'hourly')
    assert data.get_retry_queue() == []


def test_ingest_revisits_retryable_job(monkeypatch, tmp_path):
    queue_path = tmp_path / 'retry_queue.json'
    state_path = tmp_path / 'rate_state.json'
    monkeypatch.setattr(data, 'RETRY_QUEUE_PATH', queue_path)
    monkeypatch.setattr(data, 'RATE_STATE_PATH', state_path)
    monkeypatch.setattr(data, 'CACHE_DIR', tmp_path)

    calls = {'count': 0}

    def fake_load(coin_id, interval, **kwargs):
        calls['count'] += 1
        if calls['count'] == 1:
            data.enqueue_retry(coin_id, interval, reason='429', attempt=1, not_before=time.time())
            raise data.RateLimitError('429', retry_after=0)
        frame = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=40, freq='h', tz='UTC'),
            'open': [1.0] * 40,
            'high': [1.1] * 40,
            'low': [0.9] * 40,
            'close': [1.0] * 40,
            'volume': [1.0] * 40,
        })
        return frame, False

    monkeypatch.setattr(data, 'load_candles', fake_load)
    result = data.ingest_jobs([('bitcoin', 'hourly')], max_passes=3)
    assert result['completed']
    assert calls['count'] >= 2
    assert result['pending_retry_queue'] == []

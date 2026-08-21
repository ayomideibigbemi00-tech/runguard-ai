import time
import pandas as pd
from app.services import data


def test_all_jobs_revisited_after_first_429(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'RETRY_QUEUE_PATH', tmp_path / 'retry_queue.json')
    monkeypatch.setattr(data, 'RATE_STATE_PATH', tmp_path / 'rate_state.json')
    monkeypatch.setattr(data, 'CACHE_DIR', tmp_path)

    jobs = [('bitcoin', 'hourly'), ('ethereum', 'hourly'), ('solana', 'daily'), ('dogecoin', 'daily')]
    attempts = {job: 0 for job in jobs}

    def fake_load(coin_id, interval, **kwargs):
        job = (coin_id, interval)
        attempts[job] += 1
        if attempts[job] == 1:
            data.enqueue_retry(coin_id, interval, reason='429', attempt=1, not_before=time.time())
            raise data.RateLimitError('429', retry_after=0)
        frame = pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=40, freq='h', tz='UTC'),
                              'open': [1.0] * 40, 'high': [1.1] * 40, 'low': [0.9] * 40,
                              'close': [1.0] * 40, 'volume': [1.0] * 40})
        return frame, False

    monkeypatch.setattr(data, 'load_candles', fake_load)
    result = data.ingest_jobs(jobs, max_passes=5)
    assert len(result['completed']) == len(jobs)
    assert not result['pending_retry_queue']
    assert all(count == 2 for count in attempts.values())

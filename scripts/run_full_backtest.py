from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import COINS, HORIZONS
from app.services.backtest import walk_forward_backtest
from app.services.data import cache_info, get_retry_queue, ingest_jobs


def flatten(result: dict) -> dict:
    row = {k: v for k, v in result.items() if k != 'metrics'}
    row.update(result['metrics'])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description='Run Runguard real-data walk-forward backtests.')
    parser.add_argument('--coins', nargs='*', help='CoinGecko IDs or symbols; defaults to all 50 coins.')
    parser.add_argument('--intervals', nargs='*', choices=['hourly', 'daily'], default=['hourly', 'daily'])
    parser.add_argument('--refit-hourly', type=int, default=24)
    parser.add_argument('--refit-daily', type=int, default=7)
    parser.add_argument('--test-fraction', type=float, default=0.20)
    parser.add_argument('--output', default='data/backtests/runguard_backtest.csv')
    parser.add_argument('--manifest', default='data/backtests/runguard_backtest_manifest.json')
    parser.add_argument('--max-ingest-passes', type=int, default=50)
    parser.add_argument('--force-refresh', action='store_true', help='Refresh even fresh caches. This consumes API calls.')
    args = parser.parse_args()

    symbol_to_id = {c['symbol'].lower(): c['id'] for c in COINS}
    selected = []
    if args.coins:
        for value in args.coins:
            key = value.lower()
            selected.append(symbol_to_id.get(key, key))
    else:
        selected = [c['id'] for c in COINS]

    allowed = {c['id'] for c in COINS}
    unknown = [cid for cid in selected if cid not in allowed]
    if unknown:
        raise SystemExit(f'Unknown coins: {", ".join(unknown)}')

    out = ROOT / args.output
    manifest_path = ROOT / args.manifest
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(coin_id, interval) for coin_id in selected for interval in args.intervals]
    started = time.time()
    print(f'Runguard real-data backtest: {len(selected)} coins × {len(args.intervals)} intervals')
    print('Synthetic fallback is DISABLED for this audit.')
    print('CoinGecko data is ingested once into the local cache; the backtester reads local data.')
    print(f'Output: {out}')

    ingestion = ingest_jobs(jobs, force_refresh=args.force_refresh, max_passes=args.max_ingest_passes)
    pending_keys = {(q['coin_id'], q['interval']) for q in ingestion['pending_retry_queue']}
    print(f"Ingestion complete: {len(ingestion['completed'])}/{len(jobs)} jobs cached")
    if pending_keys:
        print('Still pending retry queue:')
        for q in ingestion['pending_retry_queue']:
            print(f"  {q['coin_id']} {q['interval']} retry_after={q['not_before']:.0f} reason={q['reason']}")

    # Only backtest jobs with a successful real cache. This prevents a 429 or
    # network outage from being silently converted into a synthetic or incomplete result.
    rows = []
    failures = list(ingestion['failures_observed'])
    for coin_id in selected:
        for interval in args.intervals:
            if (coin_id, interval) in pending_keys:
                continue
            info = cache_info(coin_id, interval)
            if not info.get('exists'):
                failures.append({'coin_id': coin_id, 'interval': interval, 'stage': 'preflight', 'error': 'Real cache missing.'})
                continue
            for horizon in HORIZONS[interval]:
                print(f'Backtest {coin_id} {interval} horizon={horizon}...', flush=True)
                try:
                    result = walk_forward_backtest(
                        coin_id,
                        interval,
                        horizon,
                        test_fraction=args.test_fraction,
                        refit_every=args.refit_hourly if interval == 'hourly' else args.refit_daily,
                        require_real_data=True,
                    )
                    rows.append(flatten(result))
                    print(f"  strategy={result['strategy']} MAE={result['metrics']['mae_pct']:.4f}% vs baseline={result['metrics']['baseline_mae_pct']:.4f}%")
                except Exception as exc:
                    failures.append({'coin_id': coin_id, 'interval': interval, 'horizon': horizon, 'stage': 'backtest', 'error': str(exc)})
                    print(f'  FAILED: {exc}')

    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with out.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    expected_rows = sum(len(HORIZONS[i]) for i in args.intervals) * len(selected)
    summary = {
        'generated_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'coin_count_requested': len(selected),
        'intervals_requested': args.intervals,
        'expected_result_rows': expected_rows,
        'successful_result_rows': len(rows),
        'failure_count': len(failures),
        'failures_observed': failures,
        'pending_retry_queue': ingestion['pending_retry_queue'],
        'elapsed_seconds': round(time.time() - started, 2),
        'data_source': 'CoinGecko market data only; synthetic fallback disabled',
        'cache_strategy': 'Persistent local cache; one ingestion pass before local backtesting',
        'rate_limit_strategy': 'Sliding-window limiter + retry queue + Retry-After aware exponential backoff',
        'notes': [
            'The backtest reads local CoinGecko cache files after ingestion and does not call CoinGecko per horizon.',
            'A run with pending retry jobs is intentionally incomplete and exits nonzero.',
            'Successful downloads are written immediately so interrupted runs can resume.',
            'Hourly data is limited by CoinGecko to the documented historical window for the configured plan.',
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\n===== AUDIT SUMMARY =====')
    print(f'Successful rows: {len(rows)} / {expected_rows}')
    print(f'Failures:        {len(failures)}')
    print(f'Pending retries: {len(ingestion["pending_retry_queue"])}')
    print(f'CSV:             {out}')
    print(f'Manifest:        {manifest_path}')

    return 0 if len(rows) == expected_rows and not failures and not ingestion['pending_retry_queue'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

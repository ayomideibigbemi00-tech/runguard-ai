from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import COINS
from app.services.data import ingest_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description='Download/cache real CoinGecko market data for Runguard.')
    parser.add_argument('--coins', nargs='*', help='CoinGecko IDs or symbols; defaults to all 50 coins.')
    parser.add_argument('--intervals', nargs='*', choices=['hourly', 'daily'], default=['hourly', 'daily'])
    parser.add_argument('--max-passes', type=int, default=50)
    parser.add_argument('--force-refresh', action='store_true', help='Refresh even fresh local caches. Consumes API calls.')
    args = parser.parse_args()

    symbol_to_id = {c['symbol'].lower(): c['id'] for c in COINS}
    selected = []
    for value in (args.coins or [c['id'] for c in COINS]):
        selected.append(symbol_to_id.get(value.lower(), value.lower()))
    allowed = {c['id'] for c in COINS}
    unknown = [cid for cid in selected if cid not in allowed]
    if unknown:
        raise SystemExit(f'Unknown coins: {", ".join(unknown)}')

    jobs = [(coin_id, interval) for coin_id in selected for interval in args.intervals]
    print(f'Ingesting real CoinGecko data: {len(selected)} coins × {len(args.intervals)} intervals = {len(jobs)} jobs')
    print('Synthetic fallback: DISABLED')
    print('Successful datasets are saved immediately. 429/transient failures are persisted and retried.')

    result = ingest_jobs(jobs, force_refresh=args.force_refresh, max_passes=args.max_passes)
    print(f"Completed: {len(result['completed'])}/{len(jobs)}")
    print(f"Retry queue: {len(result['pending_retry_queue'])}")

    if result['pending_retry_queue']:
        print('\nPending jobs will be retried by the next run:')
        for item in result['pending_retry_queue']:
            print(f"  {item['coin_id']} {item['interval']} | not before {item['not_before']:.0f} | {item['reason']}")
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

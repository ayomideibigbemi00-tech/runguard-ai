# Runguard AI MVP Rebuild

A website-first 50-coin crypto prediction MVP with:

- FastAPI backend
- Separate Home / Model / Predict / About pages
- Four themes: light, dark, rose, amber
- Exactly 50 selectable crypto assets
- Hourly and daily market-chart data
- User-selected coin, candle interval, and prediction horizon
- Immediate browser → API → model → result workflow
- Neural network implemented from scratch with NumPy
- 24-hour persistent local market-data cache
- Persistent CoinGecko retry queue for 429/transient failures
- Sliding-window request limiter with configurable RPM
- Retry-After aware exponential backoff
- Strict real-data ingestion mode with synthetic fallback disabled

## Run on Windows

1. Create/activate a virtual environment.
2. `pip install -r requirements.txt`
3. Optional: set `COINGECKO_API_KEY` if your CoinGecko plan requires authentication.
4. Run `run_website.bat`.
5. Open http://127.0.0.1:8765

## Important data note

The application treats CoinGecko as the external source of truth and stores successful historical downloads locally. The backtester does not call CoinGecko once per horizon; it ingests each coin/interval dataset once, then reads the local cache for all horizons.

CoinGecko's current guidance should be treated conservatively: the Public API documents 5-15 calls/minute depending on conditions, while a Demo account is documented at 30 calls/minute. Runguard therefore defaults to 10 requests/minute and allows an explicit `COINGECKO_REQUESTS_PER_MINUTE` override. Every retry goes through the same limiter, 429 responses are retried with cooldown/backoff, and retryable jobs are persisted in `data/cache/retry_queue.json`.

A 429 does not cause Runguard to abandon a coin. The ingestion queue revisits that coin after its cooldown. Successful datasets are written immediately, so an interrupted run can resume without redownloading completed jobs.


CoinGecko's current market-chart documentation supports `interval=hourly` for up to the past 100 days and `interval=daily` for daily historical data. The application uses that endpoint and constructs OHLC candles from the returned price series, avoiding dependence on the more restricted hourly/daily OHLC plan options.

## Prediction flow

The Predict page sends `{coin_id, interval, horizon}` to `POST /api/predict`. The backend loads cached/fresh candles, engineers features, trains or loads the horizon-specific NumPy model, and returns the predicted price and change.

## Files you should inspect

- `app/main.py` — FastAPI routes and API
- `app/model/network.py` — neural network and scalers
- `app/services/data.py` — CoinGecko + cache + offline fallback
- `app/services/features.py` — features and supervised windows
- `app/services/predictor.py` — model training/loading/prediction
- `app/templates/` — all website pages
- `app/static/css/style.css` — themes/design
- `app/static/js/app.js` — theme switcher and immediate prediction UI
- `tests/` — validation tests


## Accuracy and backtesting

The live predictor now uses a validation-gated ensemble. The neural network is compared with a simple recent-return baseline on a chronological validation slice. A blend weight is selected from validation data only; the final prediction therefore does not automatically trust the neural network when it is worse than a simpler method.

For research, `POST /api/backtest` performs an expanding-window, leakage-safe test on a held-out historical tail and reports MAE, RMSE, and directional accuracy for both the deployed strategy and the recent-return baseline. See `BACKTESTING.md`.

Do not interpret one backtest as proof of a trading edge. Run multiple coins, intervals, horizons, and historical periods on fresh CoinGecko data before making performance claims.


## Local live-price and prediction datasets

Runguard deliberately separates historical training data from live prediction inputs.
The model learns from historical candles in `data/cache/`, while the final monetary
prediction is anchored to the actual live CoinGecko price captured when the prediction
is created. Live-price observations are also appended locally so the observation dataset
grows as the website is used.

Live observations are persisted under `data/cache/live_prices/`:

- `live_prices.jsonl` is the append-only source-of-record for observed market prices.
- `live_price_history.csv` is a spreadsheet-friendly copy.

Every live prediction is persisted under `data/cache/predictions/`:

- `predictions.jsonl` is the source-of-record ledger.
- `prediction_history.csv` is a spreadsheet-friendly copy of the same records.

A record starts as `pending` with the exact live price and live observation timestamp captured at prediction time. Once its horizon has elapsed, Runguard resolves the record against a locally recorded live observation near the target when possible, and only uses historical market-chart data as a recovery fallback when no suitable live observation exists. It stores the actual price, actual direction, absolute error, and directional result. This makes the local history a growing labelled prediction dataset for later evaluation and analysis.

The prediction page polls the saved record while it is open, and the API resolves overdue records on prediction-history requests as well, so leaving the browser closed does not delete or lose the original prediction.

The separate candle cache remains the training/backtesting dataset. The prediction history is the growing labelled outcome dataset and can later be used to evaluate calibration and live performance without changing the historical market-data cache.

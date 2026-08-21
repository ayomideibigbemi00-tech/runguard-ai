# Runguard AI backtesting

Runguard now includes a leakage-safe walk-forward backtest. This matters because a model can look brilliant if the test period quietly leaks into preprocessing or training. That is not prediction, it is hindsight wearing a lab coat.

## What is measured

- MAE of predicted future return, in percentage points.
- RMSE of predicted future return, in percentage points.
- Directional accuracy, meaning whether the predicted return has the same sign as the realized return.
- The same three metrics for a simple baseline that predicts the mean of the most recent six known returns.

## What is held out

The final 20% of supervised samples are the test period. For each test sample, the network is trained only on samples that occurred earlier in time. Scalers are fitted inside each training window, never on the full dataset.

## Run a backtest

Start the site, then POST JSON to `/api/backtest`:

```json
{"coin_id":"bitcoin","interval":"hourly","horizon":1}
```

The API returns the metrics and whether the network beat the baseline on MAE, RMSE, and directional accuracy.

## Important interpretation rule

A single successful backtest is not proof of a durable trading edge. Run multiple coins, intervals, horizons, and historical periods. Prefer results that remain better than the baseline across out-of-sample windows.

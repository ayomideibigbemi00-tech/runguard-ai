from fastapi.testclient import TestClient
from app.main import app
from config import COINS

client = TestClient(app)


def test_pages_load():
    for path in ('/', '/model', '/predict', '/about'):
        r = client.get(path)
        assert r.status_code == 200


def test_coin_catalog_has_exactly_50_unique_ids():
    assert len(COINS) == 50
    ids = [c['id'] for c in COINS]
    assert len(ids) == len(set(ids))

    r = client.get('/api/coins')
    assert r.status_code == 200
    assert r.json()['count'] == 50


def test_api_validation_rejects_unknown_coin():
    r = client.post('/api/predict', json={'coin_id': 'not-a-coin', 'interval': 'hourly', 'horizon': 1})
    assert r.status_code == 400


def test_api_validation_rejects_bad_horizon():
    r = client.post('/api/predict', json={'coin_id': 'bitcoin', 'interval': 'hourly', 'horizon': 999})
    assert r.status_code == 400


def test_backtest_endpoint_accepts_valid_request(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, 'walk_forward_backtest', lambda coin_id, interval, horizon: {
        'coin_id': coin_id, 'interval': interval, 'horizon': horizon,
        'metrics': {'mae_pct': 1.0, 'rmse_pct': 2.0, 'directional_accuracy_pct': 50.0,
                    'baseline_mae_pct': 1.1, 'baseline_rmse_pct': 2.1,
                    'baseline_directional_accuracy_pct': 49.0,
                    'model_beats_baseline_mae': True,
                    'model_beats_baseline_rmse': True,
                    'model_beats_baseline_directional_accuracy': True},
        'interpretation': 'test'
    })
    r = client.post('/api/backtest', json={'coin_id': 'bitcoin', 'interval': 'hourly', 'horizon': 1})
    assert r.status_code == 200
    assert r.json()['metrics']['model_beats_baseline_mae'] is True


def test_data_queue_endpoint():
    response = client.get('/api/data-queue')
    assert response.status_code == 200
    body = response.json()
    assert {'pending', 'ready', 'items'} <= body.keys()


def test_xrp_alias_is_accepted(monkeypatch):
    import app.main as main
    from app.services.predictor import PredictionResult
    monkeypatch.setattr(main, 'predict', lambda coin_id, interval, horizon: PredictionResult(
        interval, horizon, '1 hour', 3.0, 3.1, 0.1, 3.33, 'Bullish',
        'CoinGecko market data', 0.1, 'hybrid', 1.0, 1.2
    ))
    r = client.post('/api/predict', json={'coin_id':'xrp','interval':'hourly','horizon':1})
    assert r.status_code == 200
    assert r.json()['coin_id'] == 'ripple'
    assert r.json()['symbol'] == 'XRP'

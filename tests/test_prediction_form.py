from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_prediction_form_returns_result(monkeypatch):
    from app.services.predictor import PredictionResult
    def fake_predict(coin_id, interval, horizon):
        return PredictionResult(interval, horizon, '1 hour', 100.0, 105.0, 5.0, 5.0, 'Bullish', 'Synthetic offline fallback', 0.1, 'hybrid', 1.0, 1.2)
    monkeypatch.setattr('app.main.predict', fake_predict)
    r = client.post('/predict', data={'coin_id':'bitcoin','interval':'hourly','horizon':'1'}, follow_redirects=False)
    assert r.status_code == 200
    assert 'Prediction completed.' in r.text
    assert 'Bitcoin (BTC)' in r.text
    assert '105.00' in r.text


def test_prediction_form_persists_prediction(monkeypatch, tmp_path):
    import app.services.predictions as predictions
    monkeypatch.setattr(predictions, 'PREDICTIONS_DIR', tmp_path)
    monkeypatch.setattr(predictions, 'PREDICTIONS_PATH', tmp_path / 'predictions.jsonl')
    monkeypatch.setattr(predictions, 'PREDICTIONS_CSV_PATH', tmp_path / 'prediction_history.csv')
    from app.services.predictor import PredictionResult
    def fake_predict(coin_id, interval, horizon):
        return PredictionResult(interval, horizon, '1 hour', 100.0, 105.0, 5.0, 5.0, 'Bullish', 'Synthetic offline fallback', 0.1, 'hybrid', 1.0, 1.2, 'abc123', '2026-08-21T10:00:00Z', '2026-08-21T11:00:00Z', 'pending')
    monkeypatch.setattr('app.main.predict', fake_predict)
    r = client.post('/predict', data={'coin_id':'bitcoin','interval':'hourly','horizon':'1'}, follow_redirects=False)
    assert r.status_code == 200
    assert '105.00' in r.text

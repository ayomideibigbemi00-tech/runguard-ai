from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import COIN_MAP, COINS, HORIZONS, FEATURES, WINDOW_SIZE
from app.services.predictor import predict
from app.services.data import load_candles, retry_queue_status, normalize_coin_id, _fetch_market_chart, fetch_current_prices
from app.services.predictions import list_predictions, prediction_summary, resolve_due_predictions
from app.services.backtest import walk_forward_backtest

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='Runguard AI', version='3.1.0')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
templates = Jinja2Templates(directory=ROOT / 'templates')


class PredictRequest(BaseModel):
    coin_id: str
    interval: str
    horizon: int


def base_context(request: Request):
    return {
        'request': request,
        'themes': ['light', 'dark', 'rose', 'amber'],
        'coins': COINS,
    }


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name='index.html', context={**base_context(request), 'page': 'home'})


@app.get('/model', response_class=HTMLResponse)
def model_page(request: Request):
    return templates.TemplateResponse(request=request, name='model.html', context={**base_context(request), 'page': 'model', 'features': FEATURES, 'window_size': WINDOW_SIZE})


@app.get('/predict', response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(request=request, name='prediction.html', context={**base_context(request), 'page': 'predict', 'horizons': HORIZONS, 'selected_coin': 'bitcoin', 'selected_interval': 'hourly', 'selected_horizon': 1})


@app.post('/predict', response_class=HTMLResponse)
def predict_form(
    request: Request,
    coin_id: str = Form(...),
    interval: str = Form(...),
    horizon: int = Form(...),
):
    context = {
        **base_context(request),
        'page': 'predict',
        'horizons': HORIZONS,
        'selected_coin': coin_id,
        'selected_interval': interval,
        'selected_horizon': horizon,
    }
    try:
        coin_id = normalize_coin_id(coin_id)
        if interval not in HORIZONS or horizon not in HORIZONS[interval]:
            raise ValueError('Invalid candle interval or prediction horizon.')
        result = predict(coin_id, interval, horizon)
        context['prediction'] = {**result.__dict__, 'coin_id': coin_id, 'coin_name': COIN_MAP[coin_id]['name'], 'symbol': COIN_MAP[coin_id]['symbol']}
    except Exception as exc:
        context['prediction_error'] = str(exc)
    return templates.TemplateResponse(request=request, name='prediction.html', context=context)


@app.get('/about', response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse(request=request, name='about.html', context={**base_context(request), 'page': 'about'})


@app.get('/api/coins')
def api_coins():
    return {'coins': COINS, 'count': len(COINS)}


@app.post('/api/predict')
def api_predict(payload: PredictRequest):
    try:
        coin_id = normalize_coin_id(payload.coin_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Unsupported coin.') from exc
    if payload.interval not in HORIZONS or payload.horizon not in HORIZONS[payload.interval]:
        raise HTTPException(status_code=400, detail='Invalid interval or horizon.')
    try:
        result = predict(coin_id, payload.interval, payload.horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    data = result.__dict__
    data['coin_id'] = coin_id
    data['coin_name'] = COIN_MAP[coin_id]['name']
    data['symbol'] = COIN_MAP[coin_id]['symbol']
    return data


@app.post('/api/backtest')
def api_backtest(payload: PredictRequest):
    try:
        coin_id = normalize_coin_id(payload.coin_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Unsupported coin.') from exc
    if payload.interval not in HORIZONS or payload.horizon not in HORIZONS[payload.interval]:
        raise HTTPException(status_code=400, detail='Invalid interval or horizon.')
    try:
        return walk_forward_backtest(coin_id, payload.interval, payload.horizon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/api/predictions')
def api_predictions(limit: int = 100, status: str | None = None, coin_id: str | None = None):
    try:
        resolved = resolve_due_predictions()
        records = list_predictions(limit=limit, status=status, coin_id=coin_id)
        return {'predictions': records, 'resolved_now': len(resolved), 'summary': prediction_summary()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get('/api/predictions/summary')
def api_predictions_summary():
    return prediction_summary()


@app.get('/api/predictions/{prediction_id}')
def api_prediction(prediction_id: str):
    resolve_due_predictions()
    records = list_predictions(limit=500)
    for record in records:
        if record.get('prediction_id') == prediction_id:
            return record
    raise HTTPException(status_code=404, detail='Prediction not found.')


@app.get('/api/data-queue')
def api_data_queue():
    """Show persisted CoinGecko jobs waiting for retry."""
    return retry_queue_status()


@app.get('/api/prices')
def api_prices():
    try:
        prices = fetch_current_prices()
        return {'prices': prices, 'count': len(prices)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get('/api/status')
def api_status():
    status = {}
    # Status samples the default Bitcoin option only, keeping this endpoint cheap.
    for interval in ('hourly', 'daily'):
        try:
            df, fallback = load_candles('bitcoin', interval)
            status[interval] = {'rows': len(df), 'source': 'synthetic' if fallback else 'coingecko'}
        except Exception as exc:
            status[interval] = {'error': str(exc)}
    return status

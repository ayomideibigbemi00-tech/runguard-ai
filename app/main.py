from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Form, Depends, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import COIN_MAP, COINS, HORIZONS, FEATURES, WINDOW_SIZE
from app.services.predictor import predict
from app.services.data import load_candles, retry_queue_status, normalize_coin_id, fetch_current_prices
from app.services.predictions import save_prediction, list_predictions, resolve_due_predictions, prediction_summary
from app.services.backtest import walk_forward_backtest
from app.db import init_db, get_db
from app.auth import hash_password, verify_password, create_session_token, get_current_user

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='Runguard AI', version='3.1.0')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
templates = Jinja2Templates(directory=ROOT / 'templates')

# Initialize database on startup
@app.on_event("startup")
def startup():
    init_db()

class PredictRequest(BaseModel):
    coin_id: str
    interval: str
    horizon: int

def base_context(request: Request, user=None):
    return {
        'request': request,
        'themes': ['light', 'dark', 'rose', 'amber'],
        'coins': COINS,
        'user': user,
    }

# Auth pages
@app.get('/signup', response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse(request=request, name='signup.html', context=base_context(request))

@app.post('/signup')
def signup(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return templates.TemplateResponse(request=request, name='signup.html', context={**base_context(request), 'error': 'Username already exists'})
    password_hash = hash_password(password)
    cur = conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    token = create_session_token(user_id)
    response = RedirectResponse(url='/', status_code=302)
    response.set_cookie("session", token, httponly=True, max_age=86400)
    return response

@app.get('/login', response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name='login.html', context=base_context(request))

@app.post('/login')
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not user or not verify_password(password, user['password_hash']):
        return templates.TemplateResponse(request=request, name='login.html', context={**base_context(request), 'error': 'Invalid username or password'})
    token = create_session_token(user['id'])
    response = RedirectResponse(url='/', status_code=302)
    response.set_cookie("session", token, httponly=True, max_age=86400)
    return response

@app.get('/logout')
def logout():
    response = RedirectResponse(url='/', status_code=302)
    response.delete_cookie("session")
    return response

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse(request=request, name='index.html', context={**base_context(request, user), 'page': 'home'})

@app.get('/market', response_class=HTMLResponse)
def market_page(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse(request=request, name='market.html', context={**base_context(request, user), 'page': 'market'})

@app.get('/history', response_class=HTMLResponse)
def history_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    resolve_due_predictions()
    return templates.TemplateResponse(request=request, name='history.html', context={**base_context(request, user), 'page': 'history'})

@app.get('/charts', response_class=HTMLResponse)
def charts_page(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse(request=request, name='charts.html', context={**base_context(request, user), 'page': 'charts'})

@app.get('/predict', response_class=HTMLResponse)
def predict_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    return templates.TemplateResponse(request=request, name='prediction.html', context={**base_context(request, user), 'page': 'predict', 'horizons': HORIZONS, 'selected_coin': 'bitcoin', 'selected_interval': 'hourly', 'selected_horizon': 1})

@app.post('/predict', response_class=HTMLResponse)
def predict_form(request: Request, coin_id: str = Form(...), interval: str = Form(...), horizon: int = Form(...)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    context = {**base_context(request, user), 'page': 'predict', 'horizons': HORIZONS, 'selected_coin': coin_id, 'selected_interval': interval, 'selected_horizon': horizon}
    try:
        coin_id = normalize_coin_id(coin_id)
        if interval not in HORIZONS or horizon not in HORIZONS[interval]:
            raise ValueError('Invalid candle interval or prediction horizon.')
        # Pass user['id'] to predict
        result = predict(coin_id, interval, horizon, user_id=user['id'])
        context['prediction'] = {**result.__dict__, 'coin_id': coin_id, 'coin_name': COIN_MAP[coin_id]['name'], 'symbol': COIN_MAP[coin_id]['symbol']}
    except Exception as exc:
        context['prediction_error'] = str(exc)
    return templates.TemplateResponse(request=request, name='prediction.html', context=context)

@app.get('/about', response_class=HTMLResponse)
def about_page(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse(request=request, name='about.html', context={**base_context(request, user), 'page': 'about'})

# API endpoints
@app.get('/api/coins')
def api_coins():
    return {'coins': COINS, 'count': len(COINS)}

@app.post('/api/predict')
def api_predict(payload: PredictRequest, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    try:
        coin_id = normalize_coin_id(payload.coin_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Unsupported coin.') from exc
    if payload.interval not in HORIZONS or payload.horizon not in HORIZONS[payload.interval]:
        raise HTTPException(status_code=400, detail='Invalid interval or horizon.')
    try:
        result = predict(coin_id, payload.interval, payload.horizon, user_id=user['id'])
        data = result.__dict__
        data['coin_id'] = coin_id
        data['coin_name'] = COIN_MAP[coin_id]['name']
        data['symbol'] = COIN_MAP[coin_id]['symbol']
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post('/api/backtest')
def api_backtest(payload: PredictRequest, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
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
def api_predictions(request: Request, limit: int = 100, status: str | None = None, coin_id: str | None = None):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    try:
        resolved = resolve_due_predictions()
        records = list_predictions(user_id=user['id'], limit=limit, status=status, coin_id=coin_id)
        return {'predictions': records, 'resolved_now': len(resolved), 'summary': prediction_summary(user_id=user['id'])}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get('/api/predictions/summary')
def api_predictions_summary(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    return prediction_summary(user_id=user['id'])

@app.get('/api/predictions/{prediction_id}')
def api_prediction(prediction_id: str, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    resolve_due_predictions()
    records = list_predictions(user_id=user['id'], limit=500)
    for record in records:
        if str(record.get('prediction_id')) == prediction_id:
            return record
    raise HTTPException(status_code=404, detail='Prediction not found.')

@app.get('/api/data-queue')
def api_data_queue():
    return retry_queue_status()

@app.get('/api/prices')
def api_prices(coin: str | None = None):
    try:
        prices = fetch_current_prices()
        if coin:
            coin = normalize_coin_id(coin)
            if coin in prices:
                return {'prices': {coin: prices[coin]}, 'count': 1}
            else:
                return {'prices': {}, 'count': 0}
        return {'prices': prices, 'count': len(prices)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get('/api/chart/{coin_id}')
def api_chart(coin_id: str):
    try:
        coin_id = normalize_coin_id(coin_id)
        df, fallback = load_candles(coin_id, 'hourly', force_refresh=False, allow_fallback=True)
        df = df.tail(100)
        labels = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        prices = df['close'].tolist()
        return {'labels': labels, 'prices': prices, 'coin_id': coin_id, 'source': 'synthetic' if fallback else 'coingecko'}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get('/api/movers')
def api_movers():
    try:
        prices = fetch_current_prices()
        if not prices:
            return {'gainers': [], 'losers': []}
        coin_list = list(prices.values())
        sorted_by_change = sorted(coin_list, key=lambda x: x.get('price_change_percentage_24h', 0), reverse=True)
        gainers = sorted_by_change[:5]
        losers = sorted_by_change[-5:]
        id_map = {v['symbol']: k for k, v in prices.items()}
        gainers = [{**coin, 'id': id_map.get(coin['symbol'], '')} for coin in gainers]
        losers = [{**coin, 'id': id_map.get(coin['symbol'], '')} for coin in losers]
        return {'gainers': gainers, 'losers': losers}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get('/api/status')
def api_status():
    status = {}
    for interval in ('hourly', 'daily'):
        try:
            df, fallback = load_candles('bitcoin', interval)
            status[interval] = {'rows': len(df), 'source': 'synthetic' if fallback else 'coingecko'}
        except Exception as exc:
            status[interval] = {'error': str(exc)}
    return status
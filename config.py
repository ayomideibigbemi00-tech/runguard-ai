from pathlib import Path
import os

# Use the Railway Volume mount point for persistent data.
DATA_DIR = Path('/app/data') if os.getenv('RAILWAY_ENVIRONMENT') else Path(__file__).resolve().parent / 'data'
CACHE_DIR = DATA_DIR / 'cache'
MODELS_DIR = DATA_DIR / 'models'

VS_CURRENCY = os.getenv('COINGECKO_VS_CURRENCY', 'usd')
COINGECKO_BASE_URL = os.getenv('COINGECKO_BASE_URL', 'https://api.coingecko.com/api/v3')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '').strip()

CACHE_EXPIRY_HOURS = float(os.getenv('CACHE_EXPIRY_HOURS', '0.1'))
REQUEST_TIMEOUT_SECONDS = float(os.getenv('REQUEST_TIMEOUT_SECONDS', '30'))

WINDOW_SIZE = 30
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

COINS = [
    {'id': 'bitcoin', 'symbol': 'BTC', 'name': 'Bitcoin'},
    {'id': 'ethereum', 'symbol': 'ETH', 'name': 'Ethereum'},
    {'id': 'tether', 'symbol': 'USDT', 'name': 'Tether'},
    {'id': 'binancecoin', 'symbol': 'BNB', 'name': 'BNB'},
    {'id': 'solana', 'symbol': 'SOL', 'name': 'Solana'},
    {'id': 'ripple', 'symbol': 'XRP', 'name': 'XRP'},
    {'id': 'usd-coin', 'symbol': 'USDC', 'name': 'USDC'},
    {'id': 'cardano', 'symbol': 'ADA', 'name': 'Cardano'},
    {'id': 'dogecoin', 'symbol': 'DOGE', 'name': 'Dogecoin'},
    {'id': 'avalanche-2', 'symbol': 'AVAX', 'name': 'Avalanche'},
    {'id': 'chainlink', 'symbol': 'LINK', 'name': 'Chainlink'},
    {'id': 'polkadot', 'symbol': 'DOT', 'name': 'Polkadot'},
    {'id': 'litecoin', 'symbol': 'LTC', 'name': 'Litecoin'},
    {'id': 'bitcoin-cash', 'symbol': 'BCH', 'name': 'Bitcoin Cash'},
    {'id': 'stellar', 'symbol': 'XLM', 'name': 'Stellar'},
    {'id': 'uniswap', 'symbol': 'UNI', 'name': 'Uniswap'},
    {'id': 'aave', 'symbol': 'AAVE', 'name': 'Aave'},
    {'id': 'monero', 'symbol': 'XMR', 'name': 'Monero'},
    {'id': 'hedera-hashgraph', 'symbol': 'HBAR', 'name': 'Hedera'},
    {'id': 'kaspa', 'symbol': 'KAS', 'name': 'Kaspa'},
]
COIN_MAP = {coin['id']: coin for coin in COINS}

HORIZONS = {
    'hourly': {
        1: '1 hour', 2: '2 hours', 3: '3 hours', 6: '6 hours', 12: '12 hours',
        24: '24 hours', 48: '2 days', 72: '3 days', 168: '7 days',
        336: '14 days', 720: '30 days'
    },
    'daily': {
        1: '1 day', 2: '2 days', 3: '3 days', 7: '7 days', 14: '14 days', 30: '30 days'
    },
}

FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'return_1', 'return_3', 'return_6',
    'sma_7', 'sma_20', 'volatility_7', 'range_pct'
]
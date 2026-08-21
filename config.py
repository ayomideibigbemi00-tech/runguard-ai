from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'
MODELS_DIR = DATA_DIR / 'models'

VS_CURRENCY = os.getenv('COINGECKO_VS_CURRENCY', 'usd')
COINGECKO_BASE_URL = os.getenv('COINGECKO_BASE_URL', 'https://api.coingecko.com/api/v3')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '').strip()

# CRITICAL: Lower this to 0.1 hours (6 minutes) to prevent 429s!
CACHE_EXPIRY_HOURS = float(os.getenv('CACHE_EXPIRY_HOURS', '0.1'))

REQUEST_TIMEOUT_SECONDS = float(os.getenv('REQUEST_TIMEOUT_SECONDS', '30'))

WINDOW_SIZE = 30
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Exactly 50 selectable CoinGecko coin IDs. The app uses IDs, not symbols, because
# CoinGecko documents IDs as the unique identifiers for coin endpoints.
COINS = [
    {'id': 'bitcoin', 'symbol': 'BTC', 'name': 'Bitcoin'},
    {'id': 'ethereum', 'symbol': 'ETH', 'name': 'Ethereum'},
    {'id': 'tether', 'symbol': 'USDT', 'name': 'Tether'},
    {'id': 'binancecoin', 'symbol': 'BNB', 'name': 'BNB'},
    {'id': 'solana', 'symbol': 'SOL', 'name': 'Solana'},
    {'id': 'usd-coin', 'symbol': 'USDC', 'name': 'USDC'},
    {'id': 'ripple', 'symbol': 'XRP', 'name': 'XRP'},
    {'id': 'dogecoin', 'symbol': 'DOGE', 'name': 'Dogecoin'},
    {'id': 'cardano', 'symbol': 'ADA', 'name': 'Cardano'},
    {'id': 'tron', 'symbol': 'TRX', 'name': 'TRON'},
    {'id': 'avalanche-2', 'symbol': 'AVAX', 'name': 'Avalanche'},
    {'id': 'chainlink', 'symbol': 'LINK', 'name': 'Chainlink'},
    {'id': 'shiba-inu', 'symbol': 'SHIB', 'name': 'Shiba Inu'},
    {'id': 'polkadot', 'symbol': 'DOT', 'name': 'Polkadot'},
    {'id': 'wrapped-bitcoin', 'symbol': 'WBTC', 'name': 'Wrapped Bitcoin'},
    {'id': 'bitcoin-cash', 'symbol': 'BCH', 'name': 'Bitcoin Cash'},
    {'id': 'litecoin', 'symbol': 'LTC', 'name': 'Litecoin'},
    {'id': 'near', 'symbol': 'NEAR', 'name': 'NEAR Protocol'},
    {'id': 'uniswap', 'symbol': 'UNI', 'name': 'Uniswap'},
    {'id': 'stellar', 'symbol': 'XLM', 'name': 'Stellar'},
    {'id': 'internet-computer', 'symbol': 'ICP', 'name': 'Internet Computer'},
    {'id': 'aptos', 'symbol': 'APT', 'name': 'Aptos'},
    {'id': 'sui', 'symbol': 'SUI', 'name': 'Sui'},
    {'id': 'filecoin', 'symbol': 'FIL', 'name': 'Filecoin'},
    {'id': 'cosmos', 'symbol': 'ATOM', 'name': 'Cosmos Hub'},
    {'id': 'vechain', 'symbol': 'VET', 'name': 'VeChain'},
    {'id': 'aave', 'symbol': 'AAVE', 'name': 'Aave'},
    {'id': 'algorand', 'symbol': 'ALGO', 'name': 'Algorand'},
    {'id': 'the-graph', 'symbol': 'GRT', 'name': 'The Graph'},
    {'id': 'arbitrum', 'symbol': 'ARB', 'name': 'Arbitrum'},
    {'id': 'optimism', 'symbol': 'OP', 'name': 'Optimism'},
    {'id': 'render-token', 'symbol': 'RENDER', 'name': 'Render'},
    {'id': 'injective-protocol', 'symbol': 'INJ', 'name': 'Injective'},
    {'id': 'maker', 'symbol': 'MKR', 'name': 'Maker'},
    {'id': 'the-sandbox', 'symbol': 'SAND', 'name': 'The Sandbox'},
    {'id': 'decentraland', 'symbol': 'MANA', 'name': 'Decentraland'},
    {'id': 'axie-infinity', 'symbol': 'AXS', 'name': 'Axie Infinity'},
    {'id': 'tezos', 'symbol': 'XTZ', 'name': 'Tezos'},
    {'id': 'eos', 'symbol': 'EOS', 'name': 'EOS'},
    {'id': 'theta-token', 'symbol': 'THETA', 'name': 'Theta Network'},
    {'id': 'flow', 'symbol': 'FLOW', 'name': 'Flow'},
    {'id': 'fantom', 'symbol': 'FTM', 'name': 'Fantom'},
    {'id': 'neo', 'symbol': 'NEO', 'name': 'Neo'},
    {'id': 'kucoin-shares', 'symbol': 'KCS', 'name': 'KuCoin Token'},
    {'id': 'wrapped-steth', 'symbol': 'WSTETH', 'name': 'Wrapped stETH'},
    {'id': 'mantle', 'symbol': 'MNT', 'name': 'Mantle'},
    {'id': 'immutable-x', 'symbol': 'IMX', 'name': 'Immutable'},
    {'id': 'hedera-hashgraph', 'symbol': 'HBAR', 'name': 'Hedera'},
    {'id': 'kaspa', 'symbol': 'KAS', 'name': 'Kaspa'},
    {'id': 'monero', 'symbol': 'XMR', 'name': 'Monero'},
]
COIN_MAP = {coin['id']: coin for coin in COINS}

# Horizon is expressed in candles. Hourly supports up to 30 days; daily supports 30 days.
HORIZONS = {
    'hourly': {
        1: '1 hour', 2: '2 hours', 3: '3 hours', 6: '6 hours', 12: '12 hours', 24: '24 hours',
        48: '2 days', 72: '3 days', 168: '7 days', 336: '14 days', 720: '30 days'
    },
    'daily': {
        1: '1 day', 2: '2 days', 3: '3 days', 7: '7 days', 14: '14 days', 30: '30 days'
    },
}

FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'return_1', 'return_3', 'return_6', 'sma_7', 'sma_20', 'volatility_7', 'range_pct'
]
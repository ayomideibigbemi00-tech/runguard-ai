from config import COINS
from app.services.data import load_candles

print("Testing all 50 coins for data availability...")
print("=" * 60)

working = []
broken = []

for coin in COINS:
    coin_id = coin['id']
    name = coin['name']
    try:
        # Try hourly data first
        df, fallback = load_candles(coin_id, 'hourly', force_refresh=True, allow_fallback=False)
        if len(df) > 0:
            working.append(coin_id)
            print(f"✅ {name} ({coin_id}) - OK ({len(df)} rows)")
        else:
            broken.append(coin_id)
            print(f"❌ {name} ({coin_id}) - No data")
    except Exception as exc:
        broken.append(coin_id)
        print(f"❌ {name} ({coin_id}) - FAILED: {exc}")

print("=" * 60)
print(f"\nWORKING: {len(working)} coins")
print(f"BROKEN: {len(broken)} coins")
print("\nBroken coins list:")
for c in broken:
    print(f"  - {c}")
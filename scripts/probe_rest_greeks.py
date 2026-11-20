#!/usr/bin/env python3
"""Quick test: does TastyTrade REST return greeks for ES futures options?"""
from dotenv import load_dotenv; load_dotenv()
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tastytrade import Session

s = Session(os.environ["TASTYTRADE_CLIENT_SECRET"], os.environ["TASTYTRADE_REFRESH_TOKEN"])
print("Auth OK")

sym = "./ESH6 E3AH6 260316C6890"

# 1. market-metrics
print("\n--- /market-metrics ---")
resp = s.sync_client.get("/market-metrics", params={"symbols": sym})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(json.dumps(resp.json(), indent=2)[:500])

# 2. market-metrics for /ES underlying
print("\n--- /market-metrics for /ES ---")
resp2 = s.sync_client.get("/market-metrics", params={"symbols": "/ES"})
print(f"Status: {resp2.status_code}")
if resp2.status_code == 200:
    data = resp2.json()
    items = data.get("data", {}).get("items", [])
    if items:
        print(json.dumps(items[0], indent=2)[:800])

# 3. Try futures-option-chains (not nested)
print("\n--- /futures-option-chains/ES ---")
resp3 = s.sync_client.get("/futures-option-chains/ES/nested")
print(f"Status: {resp3.status_code}")
if resp3.status_code == 200:
    data = resp3.json()
    # Check if any strike has greeks
    chains = data.get("data", {}).get("items", [])
    print(f"Chains: {len(chains)}")
    if chains:
        exps = chains[0].get("expirations", [])
        print(f"Expirations in first chain: {len(exps)}")

# 4. Try individual option quote via SDK
print("\n--- SDK get_option() ---")
try:
    from tastytrade.instruments import FutureOption
    opt = FutureOption.get(s, sym.strip())
    print(f"Option: {opt}")
    print(f"Fields: {[a for a in dir(opt) if not a.startswith('_')][:20]}")
except Exception as e:
    print(f"Error: {e}")

print("\nDone")

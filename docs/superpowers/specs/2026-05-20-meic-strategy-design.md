# SPX 0DTE MEIC Strategy — Design Spec

**Date:** 2026-05-20
**Source prompt:** `docs/spx_0dte_meic_early_mgmt.md`

## Goal

Add a Multiple-Entries Iron Condor (MEIC) strategy on SPX 0DTE to the
strategy-agnostic `src_v2` bot. The strategy enters every 15-20 minutes
during 11:00-15:00 ET, sizes each tranche from a configurable map,
routes the IC as two independent credit spreads, places stop-market
orders on the short legs (65-90% of total credit), scales out at
per-leg profit tiers (AM: 55/65/75/85%, PM: 65/85%), and abandons
worthless long legs.

The strategy must be:

- **Pluggable** via `bot.strategy: meic` in YAML, with no orchestrator
  changes beyond a new branch in `BotOrchestrator._create_strategy`.
- **Backtestable** via the existing `src_v2/backtest/engine.py`
  (extended, not duplicated).
- **TDD-built** — every public method covered before implementation,
  reusing existing components (`ScheduledEngine`, `PositionMonitor`,
  `BacktestEngine`, `LiveStrategy` ABC).

## Architecture summary

```
BotOrchestrator
  └── MeicStrategy.create_engine(deps)  →  ScheduledEngine
        │
        ├── ScheduledEngine.run_cycle()
        │     ├── candle/IV/large-move/stability guards (unchanged)
        │     ├── AnomalyDetector, RegimeClassifier (unchanged)
        │     ├── RiskManager.can_open_new_trade (unchanged)
        │     ├── strategy.search_candidates(chain, spot)
        │     ├── strategy.validate(setup, chain)
        │     └── ToolHandlers.place_order(setup.build_order())
        │           └── place_two_spread_ic — submits PUT + CALL credit
        │             spreads as separate orders, then stop-market on shorts
        │
        └── PositionMonitor._check_trade
              ├── strategy.check_exit(trade, quotes)
              │     ├── AM tiers @ 55/65/75/85% — close 25% per tier
              │     ├── PM tiers @ 65/85% — close 50% per tier
              │     ├── Stop trigger when short ask ≥ short_stop_price
              │     └── Abandon long when bid < 0.05
              └── honours ExitAction.leg_quantities for partial closes
```

The shared ABC (`LiveStrategy`) gains nothing; the only widening is
two new optional fields on `ExitAction`.

## Files added / changed

| Path | Action | Purpose |
|---|---|---|
| `src_v2/strategies/meic.py` | **new** | `MeicStrategy(LiveStrategy)` |
| `src_v2/strategies/base.py` | **edit** | Add `ExitAction.quantity_fraction`, `leg_quantities` |
| `src_v2/main.py` | **edit** | Route `bot.strategy == "meic"` |
| `src_v2/services/position_monitor.py` | **edit** | Honour `exit_action.leg_quantities` in `_close_leg` |
| `src_v2/agent/tool_handlers.py` | **edit** | `place_two_spread_ic` and stop-market wiring |
| `src_v2/backtest/engine.py` | **edit** | MEIC backtest path (`_process_day_meic`, `_simulate_meic_side`) |
| `config/config_meic_spx.yaml` | **new** | Strategy parameters + schedule |
| `tests_v2/test_meic_strategy.py` | **new** | Unit tests for MeicStrategy |
| `tests_v2/backtest/test_meic_backtest.py` | **new** | Backtest smoke test |

No changes to `LiveStrategy` ABC method signatures.
No new engine class.
No DB schema changes (scale-out ledger held in-strategy for v1).

## Component design

### 1. `MeicStrategy` (new)

Implements all 10 `LiveStrategy` abstract methods. Internal state:

```python
self._scaleout_state: dict[tuple[str, str], set[int]] = {}
#   key: (trade_id, side="put"|"call") → set of tier indices already fired
```

#### `configure(config)`

Reads `config.strategy.*`:

| Key | Default | Used by |
|---|---|---|
| `am_pm_cutoff` | `"12:30"` | search_candidates, check_exit |
| `am_min_credit` | `1.25` | search_candidates |
| `am_max_credit` | `1.65` | search_candidates |
| `pm_min_credit` | `0.80` | search_candidates |
| `pm_max_credit` | `1.30` | search_candidates |
| `wing_widths` | `[80, 90, 100]` | search_candidates |
| `stop_pct_of_total_credit` | `0.70` | compute_stop_loss |
| `am_scale_out_tiers` | `[0.55, 0.65, 0.75, 0.85]` | check_exit |
| `am_scale_out_fractions` | `[0.25, 0.25, 0.25, 0.25]` | check_exit |
| `pm_scale_out_tiers` | `[0.65, 0.85]` | check_exit |
| `pm_scale_out_fractions` | `[0.50, 0.50]` | check_exit |
| `abandon_long_below` | `0.05` | check_exit |
| `tranche_sizing` | `{}` | search_candidates |
| `credit_equality_threshold` | `0.20` | search_candidates ranking |

Stop-pct allowed range enforced: `0.65 ≤ stop_pct_of_total_credit ≤ 0.90`.

#### `search_candidates(chain, spot)`

1. Determine current ET clock time (use `datetime.now(ZoneInfo("America/New_York"))`).
2. Pick band: AM if `now.time() < am_pm_cutoff`, else PM.
3. For each `wing_width ∈ wing_widths`:
   - Iterate call candidates (delta does not gate MEIC — credit band does).
   - For each `sc_strike`, the candidate long call = `sc_strike + width`.
   - For each `sp_strike`, long put = `sp_strike - width`.
   - Mid both shorts and longs (`mid_price` helper).
   - `call_credit = sc_mid - lc_mid`, `put_credit = sp_mid - lp_mid`.
   - Both credits MUST fall inside the active band.
4. Stamp `qty` from `tranche_sizing.get(now.strftime("%H:%M"), default_qty)`
   onto every `OrderLeg` and `setup.extra["qty"]`.
5. Rank by `credit_diff = abs(call_credit - put_credit)`, keep top 5.

Returns the same shape as IronCondor: `{"candidates_found", "top_candidate", "candidates"}`.

#### `validate(setup, chain)`

Checks:

- Strike ordering: `long_call > short_call`, `short_put > long_put`.
- Wing symmetry: `|call_w - put_w| ≤ 5`, both `∈ [80, 100]`.
- Credit band: both `call_credit` and `put_credit` inside active band.
- `qty ≥ 1`.

Returns `{"passed": bool, "checks": [...], "failures": [...]}`.

#### `build_order(setup)`

Returns:

```python
{
    "order_type": "meic_two_spreads",
    "quantity": qty,
    "put_spread": {
        "short_strike": sp, "long_strike": lp,
        "credit": put_credit, "quantity": qty,
    },
    "call_spread": {
        "short_strike": sc, "long_strike": lc,
        "credit": call_credit, "quantity": qty,
    },
}
```

`ToolHandlers.place_two_spread_ic` consumes this and submits two
credit spreads (vertical orders), then two stop-market orders on the
shorts. Returns combined fill record with symbols populated for both
shorts and longs.

#### `compute_stop_loss(setup)`

```python
total_credit = call_credit + put_credit
stop_amount = stop_pct_of_total_credit * total_credit  # dollars per spread (×100 at broker)
return {
    "total_credit": total_credit,
    "stop_pct": stop_pct,
    "stop_amount": stop_amount,
    "short_call_stop_price": sc_mid + stop_amount,
    "short_put_stop_price":  sp_mid + stop_amount,
    "order_type": "stop_market",
}
```

#### `build_stop_orders(setup, stop_info)`

Returns:

```python
[
    {
        "side": "short_call",
        "stop_market": {"trigger_price": stop_info["short_call_stop_price"]},
    },
    {
        "side": "short_put",
        "stop_market": {"trigger_price": stop_info["short_put_stop_price"]},
    },
]
```

No stop-limit fallback (per prompt — stop-market to minimise slippage at trigger).

#### `check_exit(trade, quotes)`

Per side independently:

1. Read `entry_time_et` from trade; determine AM vs PM tier set.
2. Compute current `side_pnl_pct` from short ask (cost to buy back):
   `(credit - short_ask) / credit`.
3. For each tier index `i` in `tiers` whose pct ≤ `side_pnl_pct` and `i ∉ fired`:
   - Add `i` to `_scaleout_state[(trade_id, side)]`.
   - Close `floor(open_qty * fraction[i])` contracts of the short leg.
   - If long leg's bid ≥ `abandon_long_below`, also close
     `floor(open_qty * fraction[i])` of the long. Otherwise mark
     `extra.abandon_long_side = "call"|"put"`.
4. Stop-loss: if short ask ≥ `short_stop_price` (from trade row),
   emit `ExitAction(action="stop_loss", legs_to_close=[short, long],
   leg_quantities={short: open_qty, long: open_qty})`.

If multiple tiers cross in one bar, fire them all in the same action
(`leg_quantities` sums the fractions).

Tier list selection: if `entry_time_et.time() < am_pm_cutoff` → AM tiers,
else PM tiers. Cached on the trade dict via `extra["tier_mode"]` set
at entry time.

Returns one `ExitAction` per call; if both sides need partial closes
in the same poll, the caller calls `check_exit` once per trade and
the strategy emits a single `ExitAction` with both sides' symbols in
`legs_to_close` and `leg_quantities`.

#### `compute_trailing_stop(trade, quotes)`

Returns `None`. MEIC uses exchange-resting stop-market from entry;
no trailing logic.

#### `leg_pairs()`

```python
[("short_call_symbol", "long_call_symbol", "call"),
 ("short_put_symbol",  "long_put_symbol",  "put")]
```

#### `create_engine(deps)`

```python
from src_v2.agent.scheduled_engine import ScheduledEngine
return ScheduledEngine(deps, self)
```

### 2. `ExitAction` widening (base.py)

Add two optional fields:

```python
@dataclass
class ExitAction:
    action: str
    legs_to_close: List[str] = field(default_factory=list)
    reason: str = ""
    new_stop_price: Optional[float] = None
    # NEW:
    leg_quantities: Dict[str, int] = field(default_factory=dict)  # symbol → contracts
    quantity_fraction: float = 1.0                                # for whole-trade close
```

Defaults preserve existing IronCondor / IronFly / OrbEm behaviour
(`quantity_fraction=1.0`, empty `leg_quantities` → close all of leg).

### 3. `PositionMonitor` change

In `_check_trade`, when `exit_action.legs_to_close` is non-empty,
look up the per-leg quantity:

```python
qty = exit_action.leg_quantities.get(sym, trade.get(f"{leg}_open_qty", 1))
await self._close_leg(trade_id, sym, leg_name, reason=..., quantity=qty)
```

Extend `_close_leg` signature with `quantity: int = 1`. Pass through to
`tool_handlers.close_position(..., quantity=quantity)`.

Also: persist `open_qty` per leg on the trade row when MEIC submits.
For other strategies this stays `1` and behaviour is unchanged.

### 4. `ToolHandlers.place_two_spread_ic` (new)

For `order_type == "meic_two_spreads"`:

1. Submit PUT credit spread (sell short_put / buy long_put), qty = setup qty.
2. Submit CALL credit spread (sell short_call / buy long_call), qty = setup qty.
3. After both fill, submit stop-market on short_put and short_call legs at
   their `short_*_stop_price` levels.
4. Persist a new `Trade` row carrying both shorts, both longs, `open_qty`
   for each leg, `entry_time_et`, `tier_mode`, `short_call_stop_price`,
   `short_put_stop_price`, `call_credit`, `put_credit`, `total_credit`.

In `dry_run`/`manual` modes: simulate fills as `setup` mids (existing
pattern in `place_iron_condor`).

### 5. Backtest extension (`backtest/engine.py`)

Add a `strategy_type` switch at `BacktestEngine.__init__`:

```python
self._strategy_kind = config.get("bot", {}).get("strategy", "orb_em_ic")
```

If `strategy_kind == "meic"`, `run()` calls `_process_day_meic(day)`
instead of the existing ORB-flavoured `_process_day`. The MEIC day flow:

1. For each `tranche_time` in `entry.schedule`:
   - Build chain from CBBO at `tranche_time`.
   - Derive spot from CBBO.
   - `setup = await strategy.search_candidates(chain, spot)["top_candidate"]`.
   - `val = await strategy.validate(setup, chain)`; skip if not passed.
   - Assign OCC symbols to legs (`_make_symbol`).
   - Build per-side trade dicts (same shape as ORB path).
   - Walk 1-min bars from `tranche_time + 1min` to `eod_force_close`:
     - For each bar:
       - Compute side debit from CBBO mid of `short - long`.
       - Build `quotes` dict via `_build_quotes` (already exists).
       - Call `strategy.check_exit(trade, quotes)`.
       - If `action == "take_profit"`: apply partial close via
         `leg_quantities`, accumulate realised P&L, decrement
         `open_qty`. When `open_qty == 0`, close the side.
       - If `action == "stop_loss"`: full close, debit = stop_amount.
     - If EOD: any remaining qty expires worthless (full credit kept).
2. Emit one `BacktestResult` per side per tranche with `qty` field
   reflecting tranche size and `realized_pnl_usd` summing all partials.

This reuses `CbboStore`, `_build_chain`, `_build_quotes`, `_make_symbol`,
`_simulate_side`'s helpers — no duplication.

### 6. Config (`config/config_meic_spx.yaml`)

```yaml
bot:
  strategy: meic
  mode: dry_run

entry:
  schedule:
    - "11:00"
    - "11:20"
    - "11:40"
    - "12:00"
    - "12:20"
    - "12:40"
    - "13:00"
    - "13:20"
    - "13:40"
    - "14:00"
    - "14:20"
    - "14:40"
    - "15:00"
  stability_fraction: 0.20
  single_candle_fraction: 0.12
  fallback_stability_pts: 12.0

strategy:
  am_pm_cutoff: "12:30"
  am_min_credit: 1.25
  am_max_credit: 1.65
  pm_min_credit: 0.80
  pm_max_credit: 1.30
  wing_widths: [80, 90, 100]
  stop_pct_of_total_credit: 0.70
  am_scale_out_tiers: [0.55, 0.65, 0.75, 0.85]
  am_scale_out_fractions: [0.25, 0.25, 0.25, 0.25]
  pm_scale_out_tiers: [0.65, 0.85]
  pm_scale_out_fractions: [0.50, 0.50]
  abandon_long_below: 0.05
  credit_equality_threshold: 0.30
  default_tranche_qty: 3
  tranche_sizing:
    "11:00": 3
    "11:20": 4
    "11:40": 4
    "12:00": 5
    "12:20": 5
    "12:40": 6
    "13:00": 6
    "13:20": 5
    "13:40": 5
    "14:00": 4
    "14:20": 4
    "14:40": 3
    "15:00": 3

risk:
  max_concurrent_trades: 14
  max_daily_risk_percent: 3.0
  max_buying_power_percent: 60.0
  take_profit_threshold: 0.05

monitor:
  poll_interval_seconds: 15
  eod_close_time: "15:50"

options:
  contract_multiplier: 100
  quantity: 1                       # unused — MEIC sizes per tranche
  slippage_per_spread: 0.10
  commission_per_contract_per_leg: 1.0
  strike_step_pts: 5
  option_root: "SPXW"
```

## TDD test plan

Tests are written **before** implementation, in this order. Each test
maps to one method or one decision branch.

### `tests_v2/test_meic_strategy.py`

1. `test_configure_reads_all_keys` — every config key lands on the instance.
2. `test_configure_clamps_stop_pct` — 0.50 raises; 0.95 raises.
3. `test_search_candidates_uses_am_band_before_cutoff` — patched clock 11:30,
   chain with prices in AM band → candidate found; chain with prices in
   PM band → 0 candidates.
4. `test_search_candidates_uses_pm_band_after_cutoff` — symmetric.
5. `test_search_candidates_stamps_tranche_qty` — `tranche_sizing["11:00"]=3`,
   patched clock 11:00 → every `OrderLeg.quantity == 3`, `setup.extra["qty"]==3`.
6. `test_search_candidates_iterates_wing_widths` — verify 80/90/100 each
   yields a candidate when CBBO supports it.
7. `test_validate_strike_order` — long_call ≤ short_call fails.
8. `test_validate_wing_symmetry` — call_w=80, put_w=90 passes (≤5 diff);
   80 vs 95 fails.
9. `test_validate_credit_band` — credits outside band fail.
10. `test_build_order_two_spreads_payload_shape` — exact dict structure.
11. `test_compute_stop_loss_math` — total_credit=2.80, stop_pct=0.70 →
    stop_amount=1.96 → short stop prices = mid + 1.96.
12. `test_build_stop_orders_are_stop_market_on_shorts_only` — 2 orders,
    both `stop_market`, one for short_put, one for short_call.
13. `test_check_exit_am_tier_fires_at_55_pct` — credit 2.80, short ask
    such that side pct = 60% → fires tier 0 (55%), closes 25% qty.
14. `test_check_exit_am_tier_does_not_refire` — second call with same
    state does not re-emit tier 0.
15. `test_check_exit_am_multiple_tiers_one_bar` — side pct jumps to 80%
    → fires tiers 0,1,2 in one action; `leg_quantities` sums 75%.
16. `test_check_exit_pm_uses_pm_tiers` — entry 13:00 trade →
    only 65/85% tiers fire.
17. `test_check_exit_stop_loss_when_short_breaches_stop_price` —
    short ask ≥ short_stop_price → `action="stop_loss"`, both legs.
18. `test_check_exit_abandons_long_when_bid_below_threshold` —
    long bid 0.03 → `legs_to_close` contains short only, action sets
    `extra["abandon_long_side"]`.
19. `test_compute_trailing_stop_returns_none` — MEIC never trails.
20. `test_leg_pairs_returns_call_and_put` — exact shape.
21. `test_create_engine_returns_scheduled_engine` — type check.

### `tests_v2/backtest/test_meic_backtest.py`

22. `test_meic_backtest_single_day_one_tranche` — synthetic CBBO that
    expires worthless → realised P&L = `credit × 100 × qty` per side.
23. `test_meic_backtest_tier_fires_then_eod` — CBBO shows tier 0 hit
    mid-day, rest expires worthless → realised P&L sums partial
    debit + remainder full credit.
24. `test_meic_backtest_stop_loss` — CBBO breaches short_stop_price →
    realised P&L = `(credit - stop_amount) × 100 × qty`.

### Integration

25. `test_main_routes_meic_strategy` — `BotOrchestrator._create_strategy("meic")`
    returns `MeicStrategy` instance configured from YAML.
26. `test_scheduled_engine_calls_meic_strategy_search` — end-to-end mock
    one cycle, verify the right strategy method is invoked.

`PositionMonitor` and `ExitAction` widening are covered indirectly by
tests 13-18 plus one direct test:

27. `test_position_monitor_honours_leg_quantities` — stub strategy emits
    `ExitAction(legs_to_close=[s], leg_quantities={s: 2})` → monitor
    calls `_close_leg(..., quantity=2)`.

## Out of scope (v1)

- Dynamic stop-pct auto-tuning ("volatility tag" — config knob only).
- DB-persisted scale-out ledger (in-memory dict; on bot restart MEIC
  resumes with cleared ledger and may re-fire tiers — acceptable for
  v1, flagged for v2).
- Leftover-long retry logic (one attempt, then abandon — per prompt).
- Live broker stop-market routing for multi-leg combos at TastyTrade
  (handled by `tool_handlers.place_two_spread_ic` — implementation may
  fall back to stop-limit if the API rejects stop-market on options;
  decision logged at trade time, no code branch needed in strategy).

## Risks / open questions

- **Tier re-fire on restart.** If the bot crashes mid-day, the
  in-memory ledger is lost; on restart the strategy could re-fire
  already-completed tiers. Mitigated by storing `last_fired_tier`
  per side on the trade row at exit time (covered by test 14).
  Actually defer to v2.
- **Backtest fill-ability of stop-market.** Real fills slip past the
  stop trigger; backtest models the stop as filling at the configured
  `stop_amount` exactly. Document as a known optimism.
- **Tranche sizing not validated against buying power per slot.**
  RiskManager checks aggregate BP; per-slot qty may push total BP
  past limits. ScheduledEngine's existing `can_open_new_trade` check
  is sufficient for v1.

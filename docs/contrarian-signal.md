# Mean Reversion / Contrarian Signal

## How It Works (Simple Version)

When a market price drops or spikes sharply but **nobody is actually trading it much**, that move is probably noise — and the price will bounce back. We bet on the bounce.

```
Price jumps 12% in 30 min
        │
        ├─ Lots of volume?  → Real news. Stay away.
        │
        └─ Low volume?      → Noise. Bet it reverts.
```

---

## Detection Pipeline

```
Every 10s tick
    │
    ▼
┌─────────────────────────────┐
│  Long Price Cache (30 min)  │
│  180 data points per market │
└─────────────┬───────────────┘
              │
              ▼
      ┌───────────────┐
      │ |move| > 8% ? │──── No ───→ Skip (normal market)
      └───────┬───────┘
              │ Yes
              ▼
      ┌───────────────────┐
      │ Check Volume Ratio │
      │ current / average  │
      └───────┬───────────┘
              │
     ┌────────┼────────────┐
     │        │            │
     ▼        ▼            ▼
  > 2.5x    1.5–2.5x    < 1.5x
  ┌─────┐  ┌────────┐  ┌────────┐
  │ SKIP│  │  WEAK  │  │ STRONG │
  │     │  │conf×0.5│  │  full  │
  └─────┘  └───┬────┘  └───┬────┘
               │            │
               └─────┬──────┘
                     ▼
            ┌─────────────────┐
            │ Compute EWMA    │
            │ (reversion      │
            │  target price)  │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Shift p toward  │
            │ EWMA × confidence│
            │ = p_contrarian  │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Feed into       │
            │ Bayesian fusion │
            │ as evidence     │
            └─────────────────┘
```

---

## Volume = Confidence

Volume ratio is the single most important filter. It tells you **why** the price moved.

```
Volume Ratio          Meaning                 Action
─────────────────────────────────────────────────────
  < 1.5x             No one is trading.      STRONG signal — noise, will revert
                     Probably a bot or
                     thin orderbook slip.

  1.5 – 2.5x         Some activity.          WEAK signal — might revert,
                     Could be minor news.     confidence halved

  > 2.5x             Informed money.          SKIP — smart traders know
                     Real event driving       something, don't fight them
                     the move.
```

---

## Confidence Formula

```
confidence = vol_confidence × move_confidence

where:
  vol_confidence  = max(0, 1 - vol_ratio / 2.5)
  move_confidence = min(1, |price_move| / 0.20)
```

**Examples:**

| Volume Ratio | Price Move | Vol Conf | Move Conf | Final Conf |
|:---:|:---:|:---:|:---:|:---:|
| 0.8x | +10% | 0.68 | 0.50 | 0.34 |
| 0.5x | +15% | 0.80 | 0.75 | 0.60 |
| 1.8x | +12% | 0.28 | 0.60 | **0.08** (weak×0.5) |
| 3.0x | +20% | — | — | **SKIP** |

---

## EWMA Reversion Target

Instead of betting the price returns to a fixed point, we compute an **Exponentially Weighted Moving Average** — a smoothed "fair price" that adapts to recent history.

```
Price
 0.65 ┤                          ╭── current (spike)
 0.60 ┤                    ╭─────╯
 0.55 ┤  ─── EWMA ───────────────────── reversion target
 0.50 ┤──────╯
      └──────────────────────────────── Time
              30 min history

We bet price moves from 0.65 back toward 0.55 (EWMA)
```

The contrarian probability is shifted toward EWMA:

```
p_contrarian = current_price + (ewma - current_price) × confidence × 0.5
```

---

## Sizing & Exits

Contrarian trades are **more conservative** than normal trades because you're fighting momentum:

```
                    Normal          Contrarian
                ─────────────────────────────────
Kelly mult.     │   1.0x          │   0.5x        │  ← half size
Take Profit     │   +20%          │   +10%        │  ← grab reversion fast
Stop Loss       │   8-30%*        │   8-25%*      │  ← volatility-based (ATR)
                ─────────────────────────────────
* SL = 2.5 × ATR / entry_price, floor 8%, cap at default
```

These are stored **per-position** in the DB (`tp_pct`, `sl_pct` columns), so the monitor applies the right thresholds to each trade.

---

## Where It Lives in Code

```
agents/math_engine.py
  │
  ├── _long_price_cache     30 min of prices (180 points) per market
  ├── _mean_reversion()     Detection + confidence + EWMA target
  │       │
  │       ▼
  ├── analyze()             Feeds p_contrarian into bayesian_update()
  │                         Tags signal with contrarian=True if dominant
  │
main.py
  │
  ├── execute_signal()      If contrarian: Kelly×0.5, TP=10%, SL=8-25% (vol-based)
  │                         Volatility SL: 2.5×ATR/entry, floor 8%, cap default
  │                         Saves tp_pct/sl_pct into position
  │
  ├── monitor_positions()   Reads per-position tp_pct/sl_pct
  │
utils/db.py
  │
  ├── positions table       tp_pct REAL, sl_pct REAL columns
  ├── get_price_history()   Query snapshots from DB (for future use)
  └── save_position()       Persists tp_pct/sl_pct
```

---

## Signal Flow (Full Picture)

```
Market tick arrives (every 10s)
    │
    ▼
┌──────────────┐
│ math_engine  │
│  .analyze()  │
└──────┬───────┘
       │
       ├──→ prospect theory      ──→ p_prospect (prior)
       ├──→ historical base rate ──→ p_history
       ├──→ volume spike         ──→ p_volume
       ├──→ time decay           ──→ p_time
       ├──→ price momentum       ──→ p_momentum
       ├──→ MEAN REVERSION (new) ──→ p_contrarian  ← YOU ARE HERE
       │
       ▼
  bayesian_update(p_prospect, evidence...)
       │
       ▼
  calibration correction
       │
       ▼
  EV / KL / Kelly filters
       │
       ▼
  Signal { contrarian: true/false }
       │
       ▼
  execute_signal()
       │
       ├── contrarian=true  → Kelly×0.5, TP:10%, SL:25%
       └── contrarian=false → Kelly×1.0, TP:20%, SL:50%
```

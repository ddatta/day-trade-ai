#!/usr/bin/env python3
from __future__ import annotations
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

# Fixed absolute-date pair universe + unadjusted cache
PAIRS_CSV = ROOT / 'data' / 'drk_live_trades_updated_pairs.csv'
CACHE_CSV = ROOT / 'data' / 'massive_price_cache_unadjusted.csv'

OUT_SUMMARY = RESULTS / 'drk_3strategy_fixedpairs_redclose_summary.csv'
OUT_BASE = RESULTS / 'drk_base_fixedpairs_redclose_trade_log.csv'
OUT_WR = RESULTS / 'drk_best_winrate_fixedpairs_redclose_trade_log.csv'
OUT_NP = RESULTS / 'drk_best_netpnl_fixedpairs_redclose_trade_log.csv'

RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)
ENTRY_START = dtime(9, 31)
ENTRY_END = dtime(10, 30)

INIT_CAP = 3000.0
POS_PCT = 0.80
COMM = 0.0005
SLIP = 0.0002

CONFIGS = {
    'Base': {
        'stop': 0.04,
        'surge_arm': 0.05,
        'vol_mult': 1.5,
        'max_run_open': 50.0,
        'red_lookback': 4,
        'min_red_count': 2,
        'use_lowvol_pullback': 1,
        'min_entry_price': 1.0,
    },
    'Best WinRate': {
        'stop': 0.06,
        'surge_arm': 0.03,
        'vol_mult': 1.7,
        'max_run_open': 35.0,
        'red_lookback': 4,
        'min_red_count': 1,
        'use_lowvol_pullback': 1,
        'min_entry_price': 1.0,
    },
    'Best NetPnL': {
        'stop': 0.06,
        'surge_arm': 0.07,
        'vol_mult': 1.7,
        'max_run_open': 80.0,
        'red_lookback': 4,
        'min_red_count': 1,
        'use_lowvol_pullback': 1,
        'min_entry_price': 1.0,
    },
}

@dataclass
class Bar:
    dt: datetime
    o: float
    h: float
    l: float
    c: float
    v: float


def load_pairs(path: Path):
    rows = []
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            rows.append({'date': r['date'], 'ticker': r['ticker'], 'live_outcome': r.get('live_outcome', ''), 'mentions': int(r.get('mentions') or 1)})
    rows.sort(key=lambda x: (x['date'], x['ticker']))
    return rows


def load_bars(cache_csv: Path, needed: set[tuple[str, str]]):
    out = defaultdict(list)
    with cache_csv.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row['date'], row['ticker'])
            if key not in needed:
                continue
            try:
                dt = datetime.fromisoformat(row['dt_ny'])
                b = Bar(dt, float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume']))
            except Exception:
                continue
            tt = dt.timetz().replace(tzinfo=None)
            if tt < RTH_START or tt > RTH_END:
                continue
            out[key].append(b)
    for k in out:
        out[k].sort(key=lambda x: x.dt)
    return out


def ema(prev: float | None, x: float, n: int):
    if prev is None:
        return x
    a = 2.0 / (n + 1.0)
    return a * x + (1 - a) * prev


def simulate_day(bars: list[Bar], cfg: dict):
    if len(bars) < 20:
        return None

    vwap = []
    e9 = []
    csum_pv = csum_v = 0.0
    eprev = None
    for b in bars:
        tp = (b.h + b.l + b.c) / 3.0
        csum_pv += tp * b.v
        csum_v += b.v
        vwap.append((csum_pv / csum_v) if csum_v else b.c)
        eprev = ema(eprev, b.c, 9)
        e9.append(eprev)

    support_touched = False
    support_reclaimed = False
    rth_open = bars[0].o
    rth_high = bars[0].h

    entry_idx = None
    for i in range(8, len(bars) - 1):
        b = bars[i]
        tt = b.dt.timetz().replace(tzinfo=None)

        rth_high = max(rth_high, b.h)
        if b.l < vwap[i]:
            support_touched = True
        if support_touched and b.c >= vwap[i]:
            support_reclaimed = True

        if not (ENTRY_START <= tt <= ENTRY_END):
            continue
        if not support_reclaimed:
            continue

        run_from_open_pct = (rth_high / rth_open - 1.0) * 100.0
        if run_from_open_pct > cfg['max_run_open']:
            continue

        prev5_avg = sum(bars[j].v for j in range(i - 5, i)) / 5.0
        if b.v < cfg['vol_mult'] * prev5_avg:
            continue

        lb = cfg['red_lookback']
        red_ct = sum(1 for j in range(i - lb, i) if bars[j].c < bars[j].o)
        if red_ct < cfg['min_red_count']:
            continue

        if cfg['use_lowvol_pullback']:
            pb_vol = sum(bars[j].v for j in range(i - lb, i)) / lb
            pre = bars[max(0, i - 2 * lb):i - lb]
            if len(pre) < 2:
                continue
            pre_vol = sum(x.v for x in pre) / len(pre)
            if not (pb_vol < pre_vol):
                continue

        signal = (b.c > b.o) and (b.c > vwap[i]) and (b.c > e9[i]) and (b.c >= cfg['min_entry_price'])
        if signal:
            entry_idx = i + 1
            break

    if entry_idx is None or entry_idx >= len(bars):
        return None

    eb = bars[entry_idx]
    entry_px = eb.o * (1.0 + SLIP)
    stop_px = entry_px * (1.0 - cfg['stop'])
    armed = False

    exit_px = bars[-1].c * (1.0 - SLIP)
    exit_dt = bars[-1].dt
    exit_reason = 'eod'

    for b in bars[entry_idx + 1:]:
        if b.l <= stop_px:
            exit_px = max(0.0, stop_px * (1.0 - SLIP))
            exit_dt = b.dt
            exit_reason = 'stop_loss'
            break
        if (not armed) and (b.h >= entry_px * (1.0 + cfg['surge_arm'])):
            armed = True
        if armed and (b.c < b.o):
            exit_px = b.c * (1.0 - SLIP)
            exit_dt = b.dt
            exit_reason = 'surge_first_red_close'
            break

    return eb.dt.isoformat(), entry_px, exit_dt.isoformat(), exit_px, exit_reason


def run_cfg(name: str, cfg: dict, pairs, bars_map):
    cap = INIT_CAP
    peak = cap
    max_dd = 0.0
    trades = wins = losses = 0
    gross_win = gross_loss = 0.0
    logs = []

    for p in pairs:
        bars = bars_map.get((p['date'], p['ticker']))
        if not bars:
            continue
        sim = simulate_day(bars, cfg)
        if sim is None:
            continue

        et, ep, xt, xp, xr = sim
        size = cap * POS_PCT
        pnl_pct = (xp / ep) - 1.0
        pnl_d = size * pnl_pct
        fee_in = size * COMM
        fee_out = size * (xp / ep) * COMM
        net_d = pnl_d - fee_in - fee_out
        cap += net_d

        outcome = 'win' if net_d > 0 else 'loss'
        trades += 1
        if outcome == 'win':
            wins += 1
            gross_win += net_d
        else:
            losses += 1
            gross_loss += -net_d

        peak = max(peak, cap)
        max_dd = max(max_dd, peak - cap)

        logs.append({
            'strategy': name,
            'date': p['date'],
            'ticker': p['ticker'],
            'live_outcome': p['live_outcome'],
            'entry_time': et,
            'entry_price': round(ep, 6),
            'exit_time': xt,
            'exit_price': round(xp, 6),
            'exit_reason': xr,
            'pnl_pct': round(pnl_pct * 100.0, 4),
            'net_pnl_dollars': round(net_d, 6),
            'trade_size': round(size, 6),
            'after_trade_capital': round(cap, 6),
            'model_outcome': outcome,
        })

    wr = (wins / trades * 100.0) if trades else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else 0.0

    return {
        'strategy': name,
        'pairs_total': len(pairs),
        'pairs_with_data': sum(1 for p in pairs if (p['date'], p['ticker']) in bars_map),
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate_pct': round(wr, 4),
        'initial_capital': INIT_CAP,
        'final_capital': round(cap, 6),
        'net_pnl_dollars': round(cap - INIT_CAP, 6),
        'max_drawdown_dollars': round(max_dd, 6),
        'profit_factor': round(pf, 6),
        'assumptions': 'unadjusted prices, fixed absolute dates, exit on first red close after surge arm, minEntryPrice=1.0',
        'params': f"stop={cfg['stop']},arm={cfg['surge_arm']},vol={cfg['vol_mult']},maxrun={cfg['max_run_open']},red_lb={cfg['red_lookback']},min_red={cfg['min_red_count']},lowvol={cfg['use_lowvol_pullback']},minPrice={cfg['min_entry_price']}",
    }, logs


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    pairs = load_pairs(PAIRS_CSV)
    bars_map = load_bars(CACHE_CSV, {(p['date'], p['ticker']) for p in pairs})

    summary = []
    logs = {}
    for name, cfg in CONFIGS.items():
        s, lg = run_cfg(name, cfg, pairs, bars_map)
        summary.append(s)
        logs[name] = lg

    write_csv(OUT_SUMMARY, summary)
    write_csv(OUT_BASE, logs['Base'])
    write_csv(OUT_WR, logs['Best WinRate'])
    write_csv(OUT_NP, logs['Best NetPnL'])

    print('wrote', OUT_SUMMARY)
    print('wrote', OUT_BASE)
    print('wrote', OUT_WR)
    print('wrote', OUT_NP)
    for s in summary:
        print(s)

if __name__ == '__main__':
    main()

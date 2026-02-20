# day-trade-ai

TradingView Pine indicator + strategy and reproducible backtests for Drk WinPattern VWAP BullFlag.

## Included
- `pine/Drk_WinPattern_VWAP_BullFlag_Indicator.pine`
- `pine/Drk_WinPattern_VWAP_BullFlag_Strategy.pine`
- `scripts/run_backtest.py`
- `data/drk_live_trades_updated_pairs.csv`
- `data/massive_price_cache_unadjusted.csv`
- `results/*.csv`

## Backtest assumptions
- Unadjusted minute bars from Massive (`adjusted=false`)
- Fixed absolute ticker-date pairs from `data/drk_live_trades_updated_pairs.csv`
- Entry window 09:31-10:30 ET
- Exit on first red close after surge-arm (plus stop-loss)
- Rolling capital model: start 3000, position size 80% of current capital
- Commission 0.05%, slippage 0.02%

## Run
```bash
python3 scripts/run_backtest.py
```

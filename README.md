# day-trade-ai

TradingView Pine indicator + strategy and reproducible backtests for Drako WinPattern VWAP BullFlag.

## Included
- `pine/Drako_WinPattern_VWAP_BullFlag_Indicator.pine`
- `pine/Drako_WinPattern_VWAP_BullFlag_Strategy.pine`
- `scripts/run_backtest.py`
- `results/*.csv`

## Backtest assumptions
- Unadjusted minute bars from Massive (`adjusted=false`)
- Fixed absolute ticker-date pairs from `live_trades_updated_pairs.csv`
- Entry window 09:31-10:30 ET
- Exit on first red close after surge-arm (plus stop-loss)
- Rolling capital model: start 3000, position size 80% of current capital
- Commission 0.05%, slippage 0.02%

## Run
```bash
python3 scripts/run_backtest.py
```

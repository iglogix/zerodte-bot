# tests/dc_time_machine/test_run_backtest.py
import csv
from datetime import date

from src_v2.backtest.dc_time_machine.simulator import TradeResult
from src_v2.backtest.dc_time_machine.run_backtest import write_trades_csv, summarize


def test_write_trades_csv_roundtrips(tmp_path):
    trades = [
        TradeResult(session=date(2026, 2, 2), front=date(2026, 2, 11), back=date(2026, 2, 12),
                    k_put=4980.0, k_call=5020.0, outcome="transformed", entry_debit=1.4,
                    credit=60.8, bank=59.4, transform_minute="09:42", settlement=5000.0, pnl=515.0),
        TradeResult(session=date(2026, 2, 3), front=date(2026, 2, 12), back=date(2026, 2, 13),
                    k_put=4970.0, k_call=5030.0, outcome="closed_eod", entry_debit=1.5, pnl=-20.0),
    ]
    out = tmp_path / "trades.csv"
    write_trades_csv(trades, out)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert rows[0]["outcome"] == "transformed"
    assert float(rows[0]["pnl"]) == 515.0


def test_summarize_computes_winrate_and_pf():
    trades = [
        TradeResult(session=date(2026, 2, 2), front=date(2026, 2, 11), back=date(2026, 2, 12),
                    k_put=0, k_call=0, outcome="transformed", pnl=515.0),
        TradeResult(session=date(2026, 2, 3), front=date(2026, 2, 12), back=date(2026, 2, 13),
                    k_put=0, k_call=0, outcome="closed_eod", pnl=-20.0),
    ]
    s = summarize(trades)
    assert s["n"] == 2
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == 515.0 / 20.0

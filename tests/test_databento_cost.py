import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zerodte.backtest.databento_cost import (
    estimate_request_cost,
    CostLedger,
)


def test_estimate_request_cost_calls_metadata_get_cost_with_correct_params():
    fake_client = MagicMock()
    fake_client.metadata.get_cost.return_value = 0.0023

    cost = estimate_request_cost(
        client=fake_client,
        symbols=["SPXW  250922P06445000", "SPXW  250922C06445000"],
        start="2025-09-22T13:30:00",
        end="2025-09-22T13:42:00",
    )

    assert cost == pytest.approx(0.0023)
    kwargs = fake_client.metadata.get_cost.call_args.kwargs
    assert kwargs["dataset"] == "OPRA.PILLAR"
    assert kwargs["schema"] == "cbbo-1m"
    assert kwargs["stype_in"] == "raw_symbol"
    assert kwargs["start"] == "2025-09-22T13:30:00"
    assert kwargs["end"] == "2025-09-22T13:42:00"
    assert kwargs["symbols"] == ["SPXW  250922P06445000", "SPXW  250922C06445000"]


def test_cost_ledger_appends_and_totals(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(ledger_path)
    ledger.record(date(2025, 9, 22), "live", 0.0023, n_symbols=42, n_quotes=1234)
    ledger.record(date(2025, 9, 23), "live", 0.0019, n_symbols=42, n_quotes=987)

    assert ledger.total_usd() == pytest.approx(0.0042)
    lines = ledger_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["date"] == "2025-09-22"
    assert rec["mode"] == "live"
    assert rec["cost_usd"] == pytest.approx(0.0023)
    assert rec["n_symbols"] == 42
    assert rec["n_quotes"] == 1234


def test_cost_ledger_resumes_total_from_existing_file(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps({"date": "2025-09-22", "mode": "live", "cost_usd": 0.50,
                    "n_symbols": 42, "n_quotes": 100}) + "\n"
    )
    ledger = CostLedger(ledger_path)
    assert ledger.total_usd() == pytest.approx(0.50)

    ledger.record(date(2025, 9, 23), "live", 0.25, n_symbols=42, n_quotes=200)
    assert ledger.total_usd() == pytest.approx(0.75)


def test_cost_ledger_dry_run_records_separately(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(ledger_path)
    ledger.record(date(2025, 9, 22), "dry_run", 0.10, n_symbols=42, n_quotes=0)
    # Dry-run records should NOT count toward live total
    assert ledger.total_usd(mode="live") == pytest.approx(0.0)
    assert ledger.total_usd(mode="dry_run") == pytest.approx(0.10)

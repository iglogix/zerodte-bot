#!/usr/bin/env python3
"""Extract meaningful information from bot logs, skip noise.

New log layout: logs/YYYY-MM-DD/{strategy}.log
Legacy layout:  logs/YYYY-MM-DD.log

Usage:
    python scripts/parse_log.py                          # today, all strategies
    python scripts/parse_log.py 2026-04-09               # specific date
    python scripts/parse_log.py 2026-04-09 orb_em_ic     # specific strategy
    python scripts/parse_log.py -f                       # stream today (all strategies)
    python scripts/parse_log.py -f 2026-04-09 iron_fly   # stream specific
    python scripts/parse_log.py logs/2026-04-09.log      # legacy single file
"""

import argparse
import re
import sys
import time
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Patterns that are always noise — skip these lines entirely
NOISE_PATTERNS = [
    re.compile(r"\[DEBUG\] tastytrade:"),
    re.compile(r"\[INFO\] httpx:"),
    re.compile(r"OrderFillListener: connect"),
    re.compile(r"OrderFillListener: connected"),
    re.compile(r"OrderFillListener: disconnected"),
    re.compile(r"PositionMonitor started"),
    re.compile(r"PositionMonitor stopped"),
    re.compile(r"OrderFillListener stopped"),
    re.compile(r"OrderFillListener starting"),
    re.compile(r"Database initialized"),
    re.compile(r"TastyTrade authenticated"),
    re.compile(r"market data session opened"),
    re.compile(r"Refreshed token"),
    re.compile(r"Persistent DXLink quote streamer"),
    re.compile(r"DXLink subscribed"),
    re.compile(r"\[INFO\] LiteLLM:"),
    re.compile(r"Mode: .* \(from"),
    # Stack trace internals (keep the first ERROR line, skip the traceback)
    re.compile(r"^\s+File "),
    re.compile(r"^\s+raise "),
    re.compile(r"^httpx\."),
    re.compile(r"^litellm\."),
    re.compile(r"^tastytrade\."),
    re.compile(r"^Traceback "),
    re.compile(r"^\s+$"),
    # Lines that are just continuation of tracebacks (no timestamp prefix)
    re.compile(r"^[^\d]"),
]

# Patterns to deduplicate — only show the first occurrence + count
DEDUP_PATTERNS = {
    "orb_setup_fail": re.compile(r"ORB setup failed"),
    "anomaly_fail": re.compile(r"AnomalyDetector LLM call failed"),
    "regime_fail": re.compile(r"RegimeClassifier LLM call failed"),
    "fill_listener_max": re.compile(r"max reconnect attempts reached"),
    "credit_balance": re.compile(r"credit balance is too low"),
}

# Timestamp pattern at start of log line
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) (.+)")


def is_noise(line: str) -> bool:
    for pat in NOISE_PATTERNS:
        if pat.search(line):
            return True
    return False


def is_dedup(line: str, dedup_counts: dict[str, int]) -> bool:
    for key, pat in DEDUP_PATTERNS.items():
        if pat.search(line):
            dedup_counts[key] += 1
            if dedup_counts[key] == 1:
                return False  # first occurrence — let it through
            return True
    return False


def tag_line(line: str, strategy: str) -> str:
    """Insert [strategy] after the timestamp."""
    m = TS_RE.match(line)
    if m:
        return f"{m.group(1)} [{strategy}] {m.group(2)}"
    return f"[{strategy}] {line}"


def process_line(
    line: str, strategy: str, dedup_counts: dict[str, int]
) -> str | None:
    """Return the tagged line if meaningful, None if noise."""
    line = line.rstrip()
    if not line.strip():
        return None
    if is_noise(line):
        return None
    if is_dedup(line, dedup_counts):
        return None
    return tag_line(line, strategy)


def resolve_log_files(date_str: str | None, strategy: str | None) -> list[tuple[str, Path]]:
    """Return list of (strategy_name, path) for the given date/strategy."""
    from datetime import date as dt_date

    if date_str is None:
        date_str = dt_date.today().isoformat()

    # Check if it's a direct file path (legacy or explicit)
    p = Path(date_str)
    if p.suffix == ".log" and p.exists():
        return [("bot", p)]

    # New layout: logs/YYYY-MM-DD/*.log
    day_dir = LOGS_DIR / date_str
    if day_dir.is_dir():
        files = sorted(day_dir.glob("*.log"))
        if strategy:
            files = [f for f in files if f.stem == strategy]
        return [(f.stem, f) for f in files]

    # Legacy layout: logs/YYYY-MM-DD.log
    legacy = LOGS_DIR / f"{date_str}.log"
    if legacy.exists():
        return [("bot", legacy)]

    print(f"No logs found for {date_str}")
    sys.exit(1)


def print_dedup_summary(dedup_counts: dict[str, int]) -> None:
    has_dedup = any(c > 1 for c in dedup_counts.values())
    if has_dedup:
        print("\n--- Repeated messages (collapsed) ---")
        for key, count in dedup_counts.items():
            if count > 1:
                print(f"  [{count}x] {key.replace('_', ' ')}")


def parse_batch(files: list[tuple[str, Path]]) -> None:
    dedup_counts: dict[str, int] = {k: 0 for k in DEDUP_PATTERNS}
    all_lines: list[tuple[str, str]] = []  # (sort_key, formatted_line)

    for strategy, path in files:
        with open(path) as f:
            for line in f:
                result = process_line(line, strategy, dedup_counts)
                if result is not None:
                    # Extract timestamp for sorting across files
                    m = TS_RE.match(line)
                    sort_key = m.group(1) if m else ""
                    all_lines.append((sort_key, result))

    # Sort by timestamp when merging multiple strategy files
    if len(files) > 1:
        all_lines.sort(key=lambda x: x[0])

    for _, line in all_lines:
        if "Bot initialized" in line:
            print()
        print(line)

    print_dedup_summary(dedup_counts)


def follow(files: list[tuple[str, Path]]) -> None:
    """Stream log files like tail -f, filtering noise in real time."""
    dedup_counts: dict[str, int] = {k: 0 for k in DEDUP_PATTERNS}

    # Open all files, print existing content, then stream
    handles: list[tuple[str, object]] = []
    for strategy, path in files:
        if not path.exists():
            print(f"Waiting for {path} ...", flush=True)
            while not path.exists():
                time.sleep(1)

        fh = open(path)
        # Print existing content first
        for line in fh:
            result = process_line(line, strategy, dedup_counts)
            if result is not None:
                if "Bot initialized" in result:
                    print()
                print(result, flush=True)
        handles.append((strategy, fh))

    names = ", ".join(s for s, _ in files)
    print(f"\n--- following [{names}] (Ctrl+C to stop) ---", flush=True)

    try:
        while True:
            got_line = False
            for strategy, fh in handles:
                line = fh.readline()
                if not line:
                    continue
                got_line = True
                result = process_line(line, strategy, dedup_counts)
                if result is not None:
                    if "Bot initialized" in result:
                        print()
                    print(result, flush=True)
            if not got_line:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print_dedup_summary(dedup_counts)
        print()
    finally:
        for _, fh in handles:
            fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract meaningful bot log entries")
    parser.add_argument("date", nargs="?", help="Date (YYYY-MM-DD) or log file path")
    parser.add_argument("strategy", nargs="?", help="Strategy name (e.g. orb_em_ic)")
    parser.add_argument("-f", "--follow", action="store_true", help="Stream like tail -f")
    args = parser.parse_args()

    files = resolve_log_files(args.date, args.strategy)

    if args.follow:
        follow(files)
    else:
        parse_batch(files)

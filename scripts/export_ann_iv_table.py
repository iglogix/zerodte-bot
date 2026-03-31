#!/usr/bin/env python3
"""
Export IV surface ANN predictions to a JSON lookup table.

This avoids loading PyTorch in multiprocessing workers during optimization.
The table maps (moneyness, tte_bucket, is_call) → predicted IV.

Usage:
    python scripts/export_ann_iv_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_iv_surface import IVSurfaceANN


def main():
    model_path = "models/iv_surface_ann.pt"
    out_path = "models/iv_ann_table.json"

    print(f"Loading ANN from {model_path}...")
    ann = IVSurfaceANN.load(model_path)

    # Generate grid of predictions
    # Moneyness: 0.90 to 1.10 in 0.001 steps (200 bins)
    # TTE: 0.0001 to 0.05 years in fine steps (covers 0DTE intraday)
    #       plus coarser steps up to 4.0 years for completeness
    moneyness_vals = np.arange(0.900, 1.101, 0.001)
    tte_fine = np.arange(0.0001, 0.05, 0.001)     # 0DTE range (minutes to hours)
    tte_coarse = np.arange(0.05, 4.01, 0.05)       # multi-day range
    tte_vals = np.concatenate([tte_fine, tte_coarse])

    table = {}
    for is_call in [0, 1]:
        opt_label = "put" if is_call == 0 else "call"
        entries = []
        for tte in tte_vals:
            for m in moneyness_vals:
                iv = ann.predict_iv(float(m), float(tte), is_call)
                entries.append({
                    "m": round(float(m), 4),
                    "t": round(float(tte), 5),
                    "iv": round(iv, 6),
                })
        table[opt_label] = entries
        print(f"  {opt_label}: {len(entries)} entries")

    # Also create a compact indexed version for fast lookup
    # Key: "m_t_c" -> iv  (moneyness rounded to 0.001, tte to nearest bucket, is_call)
    compact = {}
    for is_call in [0, 1]:
        opt_label = "put" if is_call == 0 else "call"
        for entry in table[opt_label]:
            key = f"{entry['m']:.3f}_{entry['t']:.4f}"
            compact.setdefault(opt_label, {})[key] = entry["iv"]

    result = {
        "moneyness_range": [0.900, 1.100],
        "moneyness_step": 0.001,
        "tte_fine_range": [0.0001, 0.05],
        "tte_fine_step": 0.001,
        "tte_coarse_range": [0.05, 4.0],
        "tte_coarse_step": 0.05,
        "compact": compact,
    }

    with open(out_path, "w") as f:
        json.dump(result, f)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"\nSaved to {out_path} ({size_mb:.1f} MB)")
    print(f"Total entries: {sum(len(v) for v in compact.values())}")


if __name__ == "__main__":
    main()

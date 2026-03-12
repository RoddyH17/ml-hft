"""
End-to-End ML-HFT Pipeline — CLI Entry Point

Roddy Huang | ML-HFT Research (2022, restructured 2026)

Usage:
    python src/pipeline.py --data data/CN_Futures_2014.01.02.csv
    python src/pipeline.py --data data/CN_Futures_2014.01.02.csv --model xgb --folds 5
    python src/pipeline.py --data data/ --model ensemble --horizon 900

Pipeline steps:
    1. Load LOB snapshots (CSV or Parquet cache)
    2. Reconstruct LOB: pivot → typed arrays
    3. Compute signals: OBI L1, OBI L3w, depth ratio, rise ratio, WAP
    4. Generate labels: binary buy/hold based on horizon forward window
    5. Build feature matrix
    6. Walk-forward model training + evaluation
    7. Print results table + feature importances
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to path (so local imports resolve when running from root)
sys.path.insert(0, str(Path(__file__).parent))

from orderbook import load_raw_lob, reconstruct_lob_snapshots, from_parquet, to_parquet
from signals import build_feature_matrix
from models import build_X_y, compare_models


def load_or_cache(csv_path: Path, cache_dir: Path | None = None) -> object:
    """Load LOB data from Parquet cache if available, else from CSV."""
    if cache_dir is None:
        cache_dir = csv_path.parent / ".cache"

    cache_path = cache_dir / (csv_path.stem + ".parquet")

    if cache_path.exists():
        print(f"[cache] Loading from Parquet: {cache_path}")
        t0 = time.perf_counter()
        lob = from_parquet(cache_path)
        print(f"[cache] Loaded in {time.perf_counter()-t0:.2f}s")
        return lob

    print(f"[load] Reading CSV: {csv_path}")
    t0 = time.perf_counter()
    raw = load_raw_lob(str(csv_path))
    print(f"[load] CSV loaded in {time.perf_counter()-t0:.2f}s  ({len(raw):,} rows)")

    print("[lob] Reconstructing LOB snapshots...")
    t1 = time.perf_counter()
    lob = reconstruct_lob_snapshots(raw, levels=3)
    print(f"[lob] Done in {time.perf_counter()-t1:.2f}s  ({lob.best_bid.shape[0]:,} ticks)")

    # Cache for future runs
    cache_dir.mkdir(parents=True, exist_ok=True)
    to_parquet(lob, str(cache_path))
    print(f"[cache] Saved to {cache_path}")

    return lob


def run_pipeline(
    data_path: str,
    model_types: list[str],
    horizon_sec: float = 900.0,
    rise_window: int = 5,
    n_folds: int = 5,
    weights: np.ndarray | None = None,
) -> None:
    path = Path(data_path)

    # Handle directory input (process all CSV files)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            print(f"[error] No CSV files found in {path}")
            sys.exit(1)
        print(f"[pipeline] Found {len(csv_files)} CSV files, using first: {csv_files[0].name}")
        path = csv_files[0]

    # ── Step 1-2: Load + reconstruct LOB ──
    lob = load_or_cache(path)

    print(f"\n[data] LOB shape: {lob.best_bid.shape[0]:,} ticks")
    print(f"[data] Price range: {lob.best_bid.min():.2f} – {lob.best_bid.max():.2f}")
    print(f"[data] Time range: {lob.timestamp_ns[0]} – {lob.timestamp_ns[-1]}")

    # ── Step 3-4: Compute signals + labels ──
    print(f"\n[signals] Computing LOB signals (horizon={horizon_sec:.0f}s, rise_window={rise_window}) ...")
    t0 = time.perf_counter()

    feats = build_feature_matrix(
        bid_price=lob.bid_price,
        ask_price=lob.ask_price,
        bid_qty=lob.bid_qty,
        ask_qty=lob.ask_qty,
        best_bid=lob.best_bid,
        best_ask=lob.best_ask,
        timestamp_sec=(lob.timestamp_ns / 1e9).astype(np.float64),
        weights=weights,
        horizon_sec=horizon_sec,
        rise_window=rise_window,
    )
    print(f"[signals] Done in {time.perf_counter()-t0:.2f}s")

    label_rate = feats.label.mean()
    print(f"[labels] Buy rate: {label_rate:.2%}  (n={feats.label.sum():,} / {len(feats.label):,})")

    # ── Step 5: Build feature matrix ──
    print("\n[features] Building X, y ...")
    X, y, feature_names = build_X_y(feats)
    print(f"[features] X shape: {X.shape}  y shape: {y.shape}")
    print(f"[features] Features: {feature_names}")

    if len(np.unique(y)) < 2:
        print("[warn] Only one class in labels — check horizon_sec parameter")
        sys.exit(1)

    # ── Step 6: Model comparison ──
    print(f"\n[model] Walk-forward validation  folds={n_folds}  models={model_types}")
    results = compare_models(X, y, feature_names, model_types=model_types, n_folds=n_folds)

    # ── Step 7: Results ──
    print("\n" + "="*55)
    print("ML-HFT WALK-FORWARD RESULTS")
    print("="*55)
    print(results.to_string(float_format=lambda x: f"{x:.4f}"))
    print("="*55)

    best_model = results["mean_auc"].idxmax()
    print(f"\nBest model: {best_model.upper()}  AUC={results.loc[best_model, 'mean_auc']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML-HFT: LOB microstructure trading signal pipeline"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to CSV data file or directory containing CSV files"
    )
    parser.add_argument(
        "--model", default="rf",
        choices=["xgb", "lgbm", "rf", "ensemble", "all"],
        help="Model type (default: rf; 'all' runs xgb+lgbm+rf)"
    )
    parser.add_argument(
        "--horizon", type=float, default=900.0,
        help="Label horizon in seconds (default: 900 = 15 minutes)"
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of walk-forward folds (default: 5)"
    )
    parser.add_argument(
        "--rise-window", type=int, default=5,
        help="Rise ratio lookback window in ticks (default: 5)"
    )
    args = parser.parse_args()

    if args.model == "all":
        model_types = ["rf", "xgb", "lgbm"]
    else:
        model_types = [args.model]

    run_pipeline(
        data_path=args.data,
        model_types=model_types,
        horizon_sec=args.horizon,
        rise_window=args.rise_window,
        n_folds=args.folds,
    )


if __name__ == "__main__":
    main()

"""
Unit tests for LOB signal computation (signals.py).

Run: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from signals import (
    compute_obi_l1,
    compute_obi_weighted,
    compute_depth_ratio,
    compute_rise_ratio,
    compute_wap,
    generate_labels,
    compute_rolling_features,
)


# ──────────────────────────────────────────────
# OBI L1
# ──────────────────────────────────────────────

def test_obi_l1_range():
    bid = np.array([100.0, 50.0, 0.0], dtype=np.float32)
    ask = np.array([100.0, 150.0, 0.0], dtype=np.float32)
    obi = compute_obi_l1(bid, ask)
    assert obi[0] == pytest.approx(0.0, abs=1e-5)   # balanced
    assert obi[1] == pytest.approx(-0.5, abs=1e-5)  # ask-heavy
    assert obi[2] == pytest.approx(0.0, abs=1e-5)   # zero total → 0


def test_obi_l1_all_bid():
    bid = np.array([200.0], dtype=np.float32)
    ask = np.array([0.0], dtype=np.float32)
    obi = compute_obi_l1(bid, ask)
    assert obi[0] == pytest.approx(1.0, abs=1e-5)


def test_obi_l1_output_shape():
    n = 1000
    bid = np.random.rand(n).astype(np.float32)
    ask = np.random.rand(n).astype(np.float32)
    obi = compute_obi_l1(bid, ask)
    assert obi.shape == (n,)
    assert np.all(obi >= -1.0) and np.all(obi <= 1.0)


# ──────────────────────────────────────────────
# OBI Weighted
# ──────────────────────────────────────────────

def test_obi_weighted_balanced():
    bid_qty = np.ones((10, 3), dtype=np.float32)
    ask_qty = np.ones((10, 3), dtype=np.float32)
    weights = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    obi = compute_obi_weighted(bid_qty, ask_qty, weights)
    np.testing.assert_allclose(obi, 0.0, atol=1e-5)


def test_obi_weighted_range():
    n, levels = 50, 3
    bid_qty = np.random.rand(n, levels).astype(np.float32)
    ask_qty = np.random.rand(n, levels).astype(np.float32)
    weights = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    obi = compute_obi_weighted(bid_qty, ask_qty, weights)
    assert obi.shape == (n,)
    assert np.all(obi >= -1.0) and np.all(obi <= 1.0)


# ──────────────────────────────────────────────
# Depth Ratio
# ──────────────────────────────────────────────

def test_depth_ratio_balanced():
    bid_qty = np.ones((5, 3), dtype=np.float32)
    ask_qty = np.ones((5, 3), dtype=np.float32)
    dr = compute_depth_ratio(bid_qty, ask_qty)
    np.testing.assert_allclose(dr, 0.5, atol=1e-5)


def test_depth_ratio_all_bid():
    bid_qty = np.ones((5, 3), dtype=np.float32)
    ask_qty = np.zeros((5, 3), dtype=np.float32)
    dr = compute_depth_ratio(bid_qty, ask_qty)
    np.testing.assert_allclose(dr, 1.0, atol=1e-5)


def test_depth_ratio_zero_returns_half():
    bid_qty = np.zeros((5, 3), dtype=np.float32)
    ask_qty = np.zeros((5, 3), dtype=np.float32)
    dr = compute_depth_ratio(bid_qty, ask_qty)
    np.testing.assert_allclose(dr, 0.5, atol=1e-5)


# ──────────────────────────────────────────────
# Rise Ratio
# ──────────────────────────────────────────────

def test_rise_ratio_monotone_up():
    bid = np.arange(100, dtype=np.float32)
    rr = compute_rise_ratio(bid, window=5)
    # After warmup, all ticks rising → rise_ratio = 1.0
    assert np.all(rr[5:] == pytest.approx(1.0, abs=1e-5))


def test_rise_ratio_monotone_down():
    bid = np.arange(100, 0, -1, dtype=np.float32)
    rr = compute_rise_ratio(bid, window=5)
    assert np.all(rr[5:] == pytest.approx(0.0, abs=1e-5))


def test_rise_ratio_warmup_zeros():
    bid = np.random.rand(50).astype(np.float32)
    rr = compute_rise_ratio(bid, window=5)
    np.testing.assert_array_equal(rr[:5], 0.0)


# ──────────────────────────────────────────────
# WAP
# ──────────────────────────────────────────────

def test_wap_equal_qty():
    n = 10
    bid_p = np.full((n, 1), 99.0, dtype=np.float32)
    ask_p = np.full((n, 1), 101.0, dtype=np.float32)
    qty   = np.ones((n, 1), dtype=np.float32)
    wap = compute_wap(bid_p, ask_p, qty, qty)
    np.testing.assert_allclose(wap, 100.0, atol=1e-4)


def test_wap_bid_heavy():
    # More bid qty → WAP closer to ask (bid_qty dominates denominator on ask side)
    bid_p = np.array([[99.0]], dtype=np.float32)
    ask_p = np.array([[101.0]], dtype=np.float32)
    bid_q = np.array([[3.0]], dtype=np.float32)
    ask_q = np.array([[1.0]], dtype=np.float32)
    # WAP = (99*1 + 101*3) / (3+1) = (99+303)/4 = 402/4 = 100.5
    wap = compute_wap(bid_p, ask_p, bid_q, ask_q)
    assert wap[0] == pytest.approx(100.5, abs=1e-4)


# ──────────────────────────────────────────────
# Generate Labels
# ──────────────────────────────────────────────

def test_labels_trivial():
    # bid is always above ask future → all 1s
    n = 100
    ts  = np.arange(n, dtype=np.float64)
    bid = np.full(n, 105.0, dtype=np.float32)
    ask = np.full(n, 100.0, dtype=np.float32)
    labels = generate_labels(bid, ask, ts, horizon_sec=10.0)
    assert labels.sum() == n


def test_labels_shape_dtype():
    n = 200
    ts  = np.arange(n, dtype=np.float64)
    bid = np.random.rand(n).astype(np.float32) + 100
    ask = bid + 0.5
    labels = generate_labels(bid, ask, ts, horizon_sec=5.0)
    assert labels.shape == (n,)
    assert labels.dtype == np.int8


# ──────────────────────────────────────────────
# Rolling Features
# ──────────────────────────────────────────────

def test_rolling_features_keys():
    arr = np.random.rand(100).astype(np.float32)
    feats = compute_rolling_features(arr, windows=[5, 10])
    assert "mean_5" in feats
    assert "std_5"  in feats
    assert "mean_10" in feats
    assert "std_10" in feats


def test_rolling_features_shape():
    n = 200
    arr = np.random.rand(n).astype(np.float32)
    feats = compute_rolling_features(arr, windows=[10, 30])
    for v in feats.values():
        assert v.shape == (n,), f"Expected ({n},), got {v.shape}"

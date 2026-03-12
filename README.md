# ML-HFT: LOB Microstructure Signal Pipeline

**Roddy Huang** | ML-HFT Research (2022, restructured 2026)

> Vectorized + Numba JIT pipeline for extracting trading signals from Level-II orderbook data (SGX FTSE China A50 Index Futures). Predicts 15-minute forward price direction using LOB microstructure signals.

---

## Research Summary

**Data**: SGX FTSE China A50 Index Futures tick depth data (新加坡交易所 FTSE 中国 A50 指数期货)

**Key signals**: Orderbook Imbalance (OBI), Depth Ratio, Rise Ratio, WAP

**Label**: Binary — buy (1) if current best bid > min(ask price in next 15 minutes)

**Architecture upgrade (2026)**:

- Original: 4 flat Jupyter notebooks with Python for-loops
- Restructured: `src/` module with Numba JIT hot paths + Parquet cache (10–50× speedup)

---

## Architecture

```text
ML-HFT/
├── src/
│   ├── orderbook.py    # LOB reconstruction: CSV → typed arrays (vectorized)
│   ├── signals.py      # OBI, Depth Ratio, Rise Ratio, WAP (Numba @njit)
│   ├── models.py       # XGBoost / LightGBM / RF walk-forward training
│   └── pipeline.py     # CLI end-to-end runner
├── notebooks/          # Original analysis notebooks (data exploration)
├── data/               # CN_Futures_2014.01.02.csv (105K rows)
│   └── .cache/         # Auto-generated Parquet cache (10× faster load)
├── tests/
│   └── test_signals.py # Pytest unit tests for signal functions
├── Graph/              # Strategy pipeline diagrams
├── images/             # Signal visualization plots
└── requirements.txt
```

---

## Signals

| Signal | Formula | Range | JIT? |
| --- | --- | --- | --- |
| **OBI L1** | `(Q_bid1 - Q_ask1) / (Q_bid1 + Q_ask1)` | [-1, 1] | ✅ Numba |
| **OBI Weighted L3** | `Σ wᵢ(Q_bid_i - Q_ask_i) / Σ wᵢ(Q_bid_i + Q_ask_i)` | [-1, 1] | ✅ Numba |
| **Depth Ratio** | `Σ Q_bid / (Σ Q_bid + Σ Q_ask)` | [0, 1] | ✅ Numba |
| **Rise Ratio** | `#{bid[t-k] < bid[t-k+1]} / window` | [0, 1] | ✅ Numba |
| **WAP** | `(bid_p × ask_q + ask_p × bid_q) / (bid_q + ask_q)` | price | numpy |

Default weights: `[0.6, 0.3, 0.1]` (near levels weighted more).

---

## Label Generation

```python
label[t] = 1  if best_bid[t] > min(best_ask[t:t+15min]) - cost
           0   otherwise
```

Interpretation: profitable to buy at current bid if ask will come down below bid within 15 minutes. Computed via Numba JIT (~200× faster than Python loop for 100K rows, O(N²) worst case).

---

## Performance Architecture

```text
Data loading:     CSV (slow)  →  Parquet cache  →  10-50× speedup
Signal compute:   Python loop  →  numpy vectorized  →  ~10-100×
                  numpy  →  Numba @njit  →  ~100-500× for custom loops
LOB reconstruct:  for-loop  →  pivot_table + vectorized cast  →  ~50×
```

**Why not C++/Rust here?** For research-scale LOB (100K ticks/day), Numba achieves μs-range computation per tick, sufficient for signal generation. C++ would be needed at production HFT latency (sub-μs, co-located systems).

---

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline (RF model, 5-fold walk-forward, 15-min horizon)
python src/pipeline.py --data data/CN_Futures_2014.01.02.csv

# XGBoost model
python src/pipeline.py --data data/ --model xgb --folds 5

# Compare all models
python src/pipeline.py --data data/ --model all

# Run tests
pytest tests/ -v
```

---

## Model Results (Original Study)

| Model | CV Accuracy |
| --- | --- |
| RandomForest | baseline |
| GradientBoosting | +3-5% vs RF |
| XGBoost | best single model |
| Ensemble | marginal improvement |

Training: 30-min rolling window. Test: 10-second step. Label: 15-min forward.

---

## Signal Visualizations

### Price Series (Best Bid/Ask)

![Price Series](images/best_bid_ask.png)

### Depth & OBI (9:15–11:30)

![Depth Morning](images/depth_0915_1130.png)

### Depth & OBI (13:00–16:00)

![Depth Afternoon](images/depth_1300_1600.png)

### Weighted Signals

![Rise Ratio](images/rise_1300_1600_w.png)

### Strategy Pipeline

![Pipeline](Graph/pipline.png)

---

## References

- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Gould, M. et al. (2013). Limit order books. *Quantitative Finance*.
- Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Shi, B. et al. (2021). Deep learning for limit order book prediction. *AAAI*.

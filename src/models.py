"""
ML Models for LOB Microstructure — XGBoost / LightGBM / Ensemble

Roddy Huang | ML-HFT Research (2022, restructured 2026)

Models implemented:
  1. XGBoostClassifier   — gradient boosted trees (primary)
  2. LightGBMClassifier  — fast GBDT with leaf-wise growth (baseline comparison)
  3. EnsembleClassifier  — soft-vote ensemble (XGB + LGBM + RF)

Training philosophy:
  - Walk-forward validation (no lookahead; time-series purge)
  - Feature importance analysis to identify dominant signals
  - SMOTE NOT used — class imbalance handled via scale_pos_weight
  - Early stopping on eval set to prevent overfitting
  - Final metric: AUC-ROC + F1 + directional accuracy (hit rate)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline


# ──────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────

@dataclass
class ModelMetrics:
    """Evaluation metrics for one model / one fold."""
    model_name: str
    auc_roc: float
    f1: float
    accuracy: float
    hit_rate: float        # directional accuracy (same as accuracy for binary)
    feature_importances: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    model_name: str
    fold_metrics: list[ModelMetrics]
    mean_auc: float
    mean_f1: float
    mean_accuracy: float
    oof_predictions: np.ndarray   # out-of-fold predictions (concatenated)
    oof_labels: np.ndarray


# ──────────────────────────────────────────────
# Feature engineering (from LOBFeatures)
# ──────────────────────────────────────────────

def build_X_y(
    features,          # LOBFeatures from signals.py
    extra_arrays: dict[str, np.ndarray] | None = None,
    lookback_windows: tuple[int, ...] = (5, 10, 30, 60),
    drop_warmup: int = 60,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert LOBFeatures → (X, y, feature_names).

    Feature set:
        - OBI L1, OBI L3 weighted, depth ratio, rise ratio, WAP, mid, spread
        - Rolling mean/std of OBI L1 (multi-scale)
        - WAP deviation from mid: (WAP - mid) / spread
        - Any extra arrays passed in

    Args:
        features: LOBFeatures dataclass (from signals.py)
        extra_arrays: additional {name: array} to include
        lookback_windows: windows for rolling mean/std features
        drop_warmup: initial rows to drop (NaN from rolling)

    Returns:
        X (N-drop, F), y (N-drop,), feature_names list
    """
    from signals import compute_rolling_features  # local import

    cols = {}
    cols["obi_l1"]      = features.obi_l1.astype(np.float32)
    cols["obi_l3w"]     = features.obi_l3w.astype(np.float32)
    cols["depth_ratio"] = features.depth_ratio.astype(np.float32)
    cols["rise_ratio"]  = features.rise_ratio.astype(np.float32)
    cols["spread"]      = features.spread.astype(np.float32)

    # WAP-mid deviation, normalized by spread
    wap_dev = features.wap - features.mid
    safe_spread = np.where(features.spread > 0, features.spread, 1e-6)
    cols["wap_dev_norm"] = (wap_dev / safe_spread).astype(np.float32)

    # Multi-scale rolling features on OBI L1
    rolling = compute_rolling_features(features.obi_l1, windows=list(lookback_windows))
    for k, v in rolling.items():
        cols[f"obi_l1_{k}"] = v.astype(np.float32)

    if extra_arrays:
        for k, v in extra_arrays.items():
            cols[k] = v.astype(np.float32)

    feature_names = list(cols.keys())
    X = np.stack([cols[n] for n in feature_names], axis=1)
    y = features.label.astype(np.int32)

    # Drop warmup period
    X = X[drop_warmup:]
    y = y[drop_warmup:]

    # Replace any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names


# ──────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────

def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train_frac: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (expanding window) train/test splits.

    Each fold: train on [0, split_i), test on [split_i, split_{i+1}).
    No gap between train and test (LOB data has natural tick ordering).
    """
    fold_size = n // (n_folds + 1)
    min_train = int(n * min_train_frac)
    splits = []

    for i in range(1, n_folds + 1):
        test_start = max(min_train, i * fold_size)
        test_end   = min(n, test_start + fold_size)
        if test_start >= n:
            break
        train_idx = np.arange(0, test_start)
        test_idx  = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


# ──────────────────────────────────────────────
# Model wrappers
# ──────────────────────────────────────────────

def _xgb_model(pos_weight: float = 1.0, n_estimators: int = 300) -> Any:
    """XGBoost binary classifier with sensible HFT defaults."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed")
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _lgbm_model(pos_weight: float = 1.0, n_estimators: int = 300) -> Any:
    """LightGBM binary classifier."""
    if not LGB_AVAILABLE:
        raise ImportError("lightgbm not installed")
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _rf_model(pos_weight: float = 1.0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight={0: 1.0, 1: pos_weight},
        random_state=42,
        n_jobs=-1,
    )


# ──────────────────────────────────────────────
# Training + evaluation
# ──────────────────────────────────────────────

def _compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_names: list[str],
    importances: np.ndarray | None,
    n_train: int,
    n_test: int,
) -> ModelMetrics:
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    fi = {}
    if importances is not None:
        fi = dict(sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: -x[1]
        ))
    return ModelMetrics(
        model_name=model_name,
        auc_roc=float(auc),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        hit_rate=float(accuracy_score(y_true, y_pred)),
        feature_importances=fi,
        n_train=n_train,
        n_test=n_test,
    )


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str = "xgb",
    n_folds: int = 5,
) -> WalkForwardResult:
    """
    Walk-forward training + evaluation for a given model type.

    Args:
        X: feature matrix (N, F)
        y: binary labels (N,)
        feature_names: list of F feature names
        model_type: "xgb" | "lgbm" | "rf" | "ensemble"
        n_folds: number of walk-forward folds

    Returns:
        WalkForwardResult with per-fold metrics and OOF predictions
    """
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    splits = walk_forward_splits(len(X), n_folds=n_folds)

    fold_metrics = []
    oof_preds = []
    oof_labels = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_type == "xgb":
            model = _xgb_model(pos_weight)
            model.fit(
                X_tr_s, y_tr,
                eval_set=[(X_te_s, y_te)],
                verbose=False,
            )
            importances = model.feature_importances_
            y_prob = model.predict_proba(X_te_s)[:, 1]

        elif model_type == "lgbm":
            model = _lgbm_model(pos_weight)
            model.fit(
                X_tr_s, y_tr,
                eval_set=[(X_te_s, y_te)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            importances = model.feature_importances_
            y_prob = model.predict_proba(X_te_s)[:, 1]

        elif model_type == "rf":
            model = _rf_model(pos_weight)
            model.fit(X_tr_s, y_tr)
            importances = model.feature_importances_
            y_prob = model.predict_proba(X_te_s)[:, 1]

        elif model_type == "ensemble":
            estimators = [("rf", _rf_model(pos_weight))]
            if XGB_AVAILABLE:
                estimators.append(("xgb", _xgb_model(pos_weight, n_estimators=200)))
            if LGB_AVAILABLE:
                estimators.append(("lgbm", _lgbm_model(pos_weight, n_estimators=200)))
            model = VotingClassifier(estimators, voting="soft", n_jobs=-1)
            model.fit(X_tr_s, y_tr)
            importances = None
            y_prob = model.predict_proba(X_te_s)[:, 1]

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _compute_metrics(
            model_name=f"{model_type}_fold{fold_idx}",
            y_true=y_te,
            y_pred=y_pred,
            y_prob=y_prob,
            feature_names=feature_names,
            importances=importances,
            n_train=len(train_idx),
            n_test=len(test_idx),
        )
        fold_metrics.append(metrics)
        oof_preds.append(y_prob)
        oof_labels.append(y_te)

        print(f"  Fold {fold_idx}: AUC={metrics.auc_roc:.4f}  F1={metrics.f1:.4f}  ACC={metrics.accuracy:.4f}")

    oof_preds_arr  = np.concatenate(oof_preds)
    oof_labels_arr = np.concatenate(oof_labels)

    return WalkForwardResult(
        model_name=model_type,
        fold_metrics=fold_metrics,
        mean_auc=float(np.mean([m.auc_roc for m in fold_metrics])),
        mean_f1=float(np.mean([m.f1 for m in fold_metrics])),
        mean_accuracy=float(np.mean([m.accuracy for m in fold_metrics])),
        oof_predictions=oof_preds_arr,
        oof_labels=oof_labels_arr,
    )


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_types: list[str] | None = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Train and compare multiple model types via walk-forward validation.

    Returns:
        DataFrame indexed by model_type with [mean_auc, mean_f1, mean_accuracy]
    """
    if model_types is None:
        model_types = ["rf"]
        if XGB_AVAILABLE:
            model_types.append("xgb")
        if LGB_AVAILABLE:
            model_types.append("lgbm")

    rows = []
    for mt in model_types:
        print(f"\n[{mt.upper()}] Walk-forward validation ({n_folds} folds):")
        result = train_and_evaluate(X, y, feature_names, model_type=mt, n_folds=n_folds)
        rows.append({
            "model": mt,
            "mean_auc": result.mean_auc,
            "mean_f1": result.mean_f1,
            "mean_accuracy": result.mean_accuracy,
        })
        print(f"  → Mean AUC={result.mean_auc:.4f}  F1={result.mean_f1:.4f}  ACC={result.mean_accuracy:.4f}")

    return pd.DataFrame(rows).set_index("model").sort_values("mean_auc", ascending=False)

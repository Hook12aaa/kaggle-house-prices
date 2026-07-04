"""Shared experiment harness for the house-prices auto-train run.

Public contract used by every experiment's run.py:

    from common import run_experiment
    run_experiment(exp_dir, arch_class, make_model, needs_scaling=False)

`make_model()` must return a fresh unfitted sklearn-compatible regressor that
trains on a numeric design matrix and predicts in log1p(SalePrice) space.

run_experiment writes, into exp_dir:
    oof.npy        out-of-fold predictions (log space), row-aligned to FOLD_INDEX
    test_pred.npy  test predictions (log space), aligned to test.csv order
    EVALUATION.json   cv_rmse (log space) + per-fold scores + metadata

The 5-fold split is seed-locked so OOF vectors from different experiments are
mutually aligned -- the Caruana ensemble depends on this.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB))

import features as _features

TARGET = "SalePrice"
ID = "Id"
N_SPLITS = 5
SEED = 42


def _raw():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    return train, test


def load_data(drop_outliers=True):
    """Return (X_train, y_log, X_test, test_ids).

    X_train/X_test are fully numeric, imputed, identically-columned design
    matrices. y_log is log1p(SalePrice). When drop_outliers is set the four
    GrLivArea>4000 training rows are removed before fitting -- they are
    documented leverage points that distort every model.
    """
    train, test = _raw()
    if drop_outliers:
        train = train[train["GrLivArea"] <= 4000].reset_index(drop=True)
    y_log = np.log1p(train[TARGET].to_numpy(dtype=float))
    test_ids = test[ID].to_numpy()

    Xtr_raw = train.drop(columns=[TARGET])
    n_tr = len(Xtr_raw)
    combined = pd.concat([Xtr_raw, test], axis=0, ignore_index=True)
    combined = _features.build_features(combined)
    combined = combined.drop(columns=[ID])

    combined = pd.get_dummies(combined, dummy_na=False)
    combined = combined.fillna(combined.median(numeric_only=True))
    combined = combined.astype(float)

    X_train = combined.iloc[:n_tr].reset_index(drop=True)
    X_test = combined.iloc[n_tr:].reset_index(drop=True)
    return X_train, y_log, X_test, test_ids


def folds(n):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n)))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_experiment(exp_dir, arch_class, make_model, needs_scaling=False,
                   drop_outliers=True, notes=""):
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    X, y, X_test, test_ids = load_data(drop_outliers=drop_outliers)
    Xv = X.to_numpy()
    Xtv = X_test.to_numpy()

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for k, (tr_idx, va_idx) in enumerate(folds(len(y))):
        if needs_scaling:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xv[tr_idx])
            Xva = sc.transform(Xv[va_idx])
            Xte = sc.transform(Xtv)
        else:
            Xtr, Xva, Xte = Xv[tr_idx], Xv[va_idx], Xtv
        model = make_model()
        model.fit(Xtr, y[tr_idx])
        p = model.predict(Xva)
        oof[va_idx] = p
        test_pred += model.predict(Xte) / N_SPLITS
        fold_scores.append(rmse(y[va_idx], p))

    cv_rmse = rmse(y, oof)
    np.save(exp_dir / "oof.npy", oof)
    np.save(exp_dir / "test_pred.npy", test_pred)
    np.save(exp_dir / "test_ids.npy", test_ids)
    np.save(exp_dir / "y_log.npy", y)

    evaluation = {
        "experiment": exp_dir.name,
        "arch_class": arch_class,
        "cv_rmse": cv_rmse,
        "fold_rmse": fold_scores,
        "fold_rmse_std": float(np.std(fold_scores)),
        "n_features": int(X.shape[1]),
        "drop_outliers": drop_outliers,
        "needs_scaling": needs_scaling,
        "n_train": int(len(y)),
        "notes": notes,
    }
    (exp_dir / "EVALUATION.json").write_text(json.dumps(evaluation, indent=2))
    print(json.dumps(evaluation, indent=2))
    return evaluation

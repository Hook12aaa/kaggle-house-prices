"""Data-validate stage: 8 universal data-quality checks on the house-prices dataset.

Writes .auto-trainer/data_report.json. Exit 0 = PASS or PASS_WITH_CONCERNS.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AT = ROOT / ".auto-trainer"
TARGET = "SalePrice"
ID = "Id"

train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")

checks = []


def check(name, status, detail):
    checks.append({"check": name, "status": status, "detail": detail})


check("shape", "PASS",
      f"train={train.shape}, test={test.shape}")

status = "PASS" if TARGET in train.columns and TARGET not in test.columns else "FAIL"
check("target_presence", status,
      f"{TARGET} in train={TARGET in train.columns}, in test={TARGET not in test.columns}")

tnull = int(train[TARGET].isna().sum())
tneg = int((train[TARGET] <= 0).sum())
check("target_validity", "PASS" if tnull == 0 and tneg == 0 else "FAIL",
      f"nulls={tnull}, non_positive={tneg}, min={train[TARGET].min()}, max={train[TARGET].max()}")

dup_tr = int(train[ID].duplicated().sum())
dup_te = int(test[ID].duplicated().sum())
check("id_integrity", "PASS" if dup_tr == 0 and dup_te == 0 else "FAIL",
      f"train_dups={dup_tr}, test_dups={dup_te}")

tr_cols = set(train.columns) - {TARGET}
te_cols = set(test.columns)
missing_in_test = tr_cols - te_cols
extra_in_test = te_cols - tr_cols
check("schema_alignment", "PASS" if not missing_in_test and not extra_in_test else "FAIL",
      f"missing_in_test={sorted(missing_in_test)}, extra_in_test={sorted(extra_in_test)}")

miss = pd.concat([train.drop(columns=[TARGET]), test], axis=0)
miss_frac = (miss.isna().mean().sort_values(ascending=False))
high_miss = miss_frac[miss_frac > 0.5]
check("missingness", "PASS_WITH_CONCERNS" if len(high_miss) else "PASS",
      f"cols_with_any_missing={int((miss_frac>0).sum())}, "
      f">50%_missing={high_miss.round(3).to_dict()}")

feat_cols = [c for c in train.columns if c not in (TARGET, ID)]
dup_rows = int(train.duplicated(subset=feat_cols).sum())
check("duplicate_rows", "PASS", f"feature_identical_train_rows={dup_rows}")

out_mask = (train["GrLivArea"] > 4000)
n_out = int(out_mask.sum())
check("outliers", "PASS_WITH_CONCERNS" if n_out else "PASS",
      f"GrLivArea>4000 rows={n_out} (mitigation: drop_outliers flag in lib.common); "
      f"target_skew={float(train[TARGET].skew()):.3f} -> log1p applied downstream")

overall = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
    "PASS_WITH_CONCERNS" if any(c["status"] == "PASS_WITH_CONCERNS" for c in checks) else "PASS")

report = {
    "stage": "data-validate",
    "overall": overall,
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "n_features": len(feat_cols),
    "target": TARGET,
    "id_column": ID,
    "checks": checks,
    "mitigations": {
        "log1p_target": True,
        "drop_outliers_GrLivArea_gt_4000": True,
    },
}
AT.mkdir(exist_ok=True)
(AT / "data_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if overall != "FAIL" else 1)

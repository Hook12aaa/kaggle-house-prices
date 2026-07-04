"""Independent validation of every experiment artifact + experiment-tree assembly.

Recomputes cv_rmse from saved oof.npy rather than trusting EVALUATION.json, and
asserts oof/test_pred shapes and cross-experiment OOF alignment (identical y_log
across experiments) so the Caruana ensemble can blend them safely.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AT = ROOT / ".auto-trainer"
EXP = AT / "experiments"
N_TRAIN = 1456
N_TEST = 1459


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


rows = []
ref_y = None
problems = []

for d in sorted(EXP.iterdir()):
    if not (d / "EVALUATION.json").exists():
        problems.append(f"{d.name}: missing EVALUATION.json")
        continue
    ev = json.loads((d / "EVALUATION.json").read_text())
    oof = np.load(d / "oof.npy")
    tp = np.load(d / "test_pred.npy")
    y = np.load(d / "y_log.npy")
    if oof.shape != (N_TRAIN,):
        problems.append(f"{d.name}: oof shape {oof.shape} != ({N_TRAIN},)")
    if tp.shape != (N_TEST,):
        problems.append(f"{d.name}: test_pred shape {tp.shape} != ({N_TEST},)")
    if ref_y is None:
        ref_y = y
    elif not np.allclose(ref_y, y):
        problems.append(f"{d.name}: y_log misaligned vs reference")
    recomputed = rmse(y, oof)
    reported = ev["cv_rmse"]
    if abs(recomputed - reported) > 1e-6:
        problems.append(f"{d.name}: cv_rmse mismatch reported={reported} recomputed={recomputed}")
    rows.append({
        "experiment": ev["experiment"],
        "arch_class": ev["arch_class"],
        "cv_rmse": recomputed,
        "fold_rmse_std": ev["fold_rmse_std"],
        "n_features": ev["n_features"],
        "status": "ACCEPT",
    })

rows.sort(key=lambda r: r["cv_rmse"])
baseline = next(r for r in rows if r["experiment"] == "exp_000_baseline")["cv_rmse"]
for r in rows:
    r["beats_baseline"] = bool(r["cv_rmse"] < baseline) or r["experiment"] == "exp_000_baseline"

classes = sorted(set(r["arch_class"] for r in rows))
tree = {
    "objective": "minimize log-space RMSE of log1p(SalePrice)",
    "baseline_cv_rmse": baseline,
    "n_experiments": len(rows),
    "architecture_classes": classes,
    "n_classes": len(classes),
    "experiments": rows,
    "validation_problems": problems,
    "global_status": "EXPLORING",
}
(AT / "experiment-tree.json").write_text(json.dumps(tree, indent=2))

print("VALIDATION PROBLEMS:", problems if problems else "NONE")
print(f"\n{'experiment':28} {'class':12} {'cv_rmse':>10} {'std':>8}  beats_base")
for r in rows:
    print(f"{r['experiment']:28} {r['arch_class']:12} {r['cv_rmse']:>10.5f} "
          f"{r['fold_rmse_std']:>8.5f}  {r['beats_baseline']}")
print(f"\nclasses={classes}  baseline={baseline:.5f}")

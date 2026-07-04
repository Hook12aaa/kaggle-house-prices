"""Generate submission.csv from the winning model and write final-report.md.

The winner is the lowest-cv_rmse node across single experiments and the ensemble.
Predictions are stored in log1p space, so SalePrice = expm1(test_pred). Validates
the submission against sample_submission.csv (row count, Id alignment, positivity)
before declaring DONE.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AT = ROOT / ".auto-trainer"
EXP = AT / "experiments"

tree = json.loads((AT / "experiment-tree.json").read_text())
ens = json.loads((EXP / "exp_ensemble" / "EVALUATION.json").read_text())

candidates = [(e["experiment"], e["cv_rmse"]) for e in tree["experiments"]]
candidates.append((ens["experiment"], ens["cv_rmse"]))
winner, winner_rmse = min(candidates, key=lambda t: t[1])

test_pred = np.load(EXP / winner / "test_pred.npy")
test_ids = np.load(EXP / winner / "test_ids.npy")
sale_price = np.expm1(test_pred)

sub = pd.DataFrame({"Id": test_ids.astype(int), "SalePrice": sale_price})

sample = pd.read_csv(ROOT / "sample_submission.csv")
assert len(sub) == len(sample), f"row count {len(sub)} != {len(sample)}"
assert list(sub["Id"]) == list(sample["Id"]), "Id order does not match sample_submission"
assert sub["SalePrice"].notna().all(), "NaN in predictions"
assert (sub["SalePrice"] > 0).all(), "non-positive prediction"

sub.to_csv(ROOT / "submission.csv", index=False)

rows = sorted(tree["experiments"], key=lambda e: e["cv_rmse"])
lines = []
lines.append("# Auto-Train Final Report — House Prices\n")
lines.append(f"**Objective:** minimize log-space RMSE of `log1p(SalePrice)` "
             f"(matches Kaggle leaderboard metric).\n")
lines.append(f"**Status:** {tree['global_status']}  |  "
             f"**Experiments:** {tree['n_experiments']}  |  "
             f"**Architecture classes:** {', '.join(tree['architecture_classes'])} "
             f"({tree['n_classes']})\n")
lines.append(f"**Baseline (exp_000 Ridge):** {tree['baseline_cv_rmse']:.5f}\n")
lines.append(f"**Winner:** `{winner}`  CV log-RMSE **{winner_rmse:.5f}**\n")

lines.append("\n## Leaderboard (5-fold CV, log-space RMSE)\n")
lines.append("| rank | experiment | class | cv_rmse | fold_std | beats baseline |")
lines.append("|---|---|---|---|---|---|")
for i, e in enumerate(rows, 1):
    lines.append(f"| {i} | {e['experiment']} | {e['arch_class']} | "
                 f"{e['cv_rmse']:.5f} | {e['fold_rmse_std']:.5f} | {e['beats_baseline']} |")
lines.append(f"| — | **exp_ensemble** | ensemble | **{ens['cv_rmse']:.5f}** | — | True |")

lines.append("\n## Caruana ensemble\n")
lines.append(f"Greedy forward selection with replacement, {ens['rounds']} rounds. "
             f"Improvement over best single (`{ens['best_single']}` "
             f"{ens['best_single_cv_rmse']:.5f}): "
             f"**{ens['improvement_vs_best_single']:.5f}**.\n")
lines.append("| component | weight |")
lines.append("|---|---|")
for k, v in sorted(ens["weights"].items(), key=lambda kv: -kv[1]):
    lines.append(f"| {k} | {v:.3f} |")

lines.append("\n## Convergence\n")
conv = tree["convergence"]
lines.append(f"Tier 2 coverage: {conv['tier2_coverage']['n_classes']} classes "
             f">= {conv['tier2_coverage']['min_required']} required → "
             f"{conv['tier2_coverage']['satisfied']}.\n")
lines.append("| class | variants | best cv_rmse | exhausted | reason |")
lines.append("|---|---|---|---|---|")
for cls, v in sorted(conv["tier1_within_class"].items()):
    lines.append(f"| {cls} | {v['n_variants']} | {v['best']:.5f} | "
                 f"{v['exhausted']} | {v['reason']} |")

lines.append("\n## Submission\n")
lines.append(f"`submission.csv` written: {len(sub)} rows, Id-aligned to "
             f"sample_submission.csv, all predictions positive. "
             f"SalePrice range ${sub['SalePrice'].min():,.0f}–${sub['SalePrice'].max():,.0f}, "
             f"median ${sub['SalePrice'].median():,.0f}.\n")

(AT / "final-report.md").write_text("\n".join(lines))

print(f"WINNER: {winner}  cv_rmse={winner_rmse:.5f}")
print(f"submission.csv: {len(sub)} rows, "
      f"SalePrice [{sub['SalePrice'].min():.0f}, {sub['SalePrice'].max():.0f}], "
      f"median {sub['SalePrice'].median():.0f}")
print("VERDICT: DONE")

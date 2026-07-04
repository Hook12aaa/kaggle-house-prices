"""Caruana (2004) greedy forward-selection ensemble over ACCEPTED experiment OOFs.

Selection minimizes OOF log-RMSE with replacement, so a strong model can be
weighted up repeatedly and weak models simply never get picked. The resulting
integer counts become the blend weights, applied identically to the seed-aligned
test predictions. Writes experiments/exp_ensemble with status ENSEMBLE so the
convergence/Pareto passes ignore it.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AT = ROOT / ".auto-trainer"
EXP = AT / "experiments"
N_ROUNDS = 100


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


tree = json.loads((AT / "experiment-tree.json").read_text())
accepted = [e["experiment"] for e in tree["experiments"] if e["status"] == "ACCEPT"]

names, oofs, tests = [], [], []
y = None
for name in accepted:
    d = EXP / name
    if name == "exp_ensemble":
        continue
    names.append(name)
    oofs.append(np.load(d / "oof.npy"))
    tests.append(np.load(d / "test_pred.npy"))
    if y is None:
        y = np.load(d / "y_log.npy")
oofs = np.vstack(oofs)
tests = np.vstack(tests)

singles = {names[i]: rmse(y, oofs[i]) for i in range(len(names))}
best_single_name = min(singles, key=singles.get)
best_single = singles[best_single_name]

counts = np.zeros(len(names), dtype=int)
ens_sum = np.zeros_like(y)
trajectory = []
for r in range(N_ROUNDS):
    best_idx, best_score = None, None
    for i in range(len(names)):
        cand = (ens_sum + oofs[i]) / (counts.sum() + 1)
        s = rmse(y, cand)
        if best_score is None or s < best_score:
            best_score, best_idx = s, i
    counts[best_idx] += 1
    ens_sum += oofs[best_idx]
    trajectory.append(best_score)
    if r >= 5 and abs(trajectory[-1] - trajectory[-2]) < 1e-7:
        break

total = counts.sum()
weights = counts / total
oof_blend = (weights[:, None] * oofs).sum(axis=0)
test_blend = (weights[:, None] * tests).sum(axis=0)
ens_rmse = rmse(y, oof_blend)

ens_dir = EXP / "exp_ensemble"
ens_dir.mkdir(exist_ok=True)
np.save(ens_dir / "oof.npy", oof_blend)
np.save(ens_dir / "test_pred.npy", test_blend)
np.save(ens_dir / "test_ids.npy", np.load(EXP / best_single_name / "test_ids.npy"))
np.save(ens_dir / "y_log.npy", y)

weight_map = {names[i]: float(weights[i]) for i in range(len(names)) if counts[i] > 0}
evaluation = {
    "experiment": "exp_ensemble",
    "arch_class": "ensemble",
    "method": "caruana_greedy_with_replacement",
    "cv_rmse": ens_rmse,
    "best_single": best_single_name,
    "best_single_cv_rmse": best_single,
    "improvement_vs_best_single": best_single - ens_rmse,
    "weights": weight_map,
    "rounds": int(total),
    "status": "ENSEMBLE",
}
(ens_dir / "EVALUATION.json").write_text(json.dumps(evaluation, indent=2))
print(json.dumps(evaluation, indent=2))

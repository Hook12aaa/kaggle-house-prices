"""Two-tier convergence detector. Reads experiment-tree.json, writes back global_status.

Tier 1 (within-class exhaustion): a class is EXHAUSTED when >=2 distinct variants
have been tried, or when its best variant is >10% worse than the global best (the
class is non-competitive and not worth further refinement).

Tier 2 (cross-class coverage): the number of explored classes meets the objective's
architecture_classes_minimum.

CONVERGED requires every explored class EXHAUSTED and coverage satisfied; otherwise
EXPLORING. Prints the verdict token.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AT = ROOT / ".auto-trainer"

MIN_CLASSES = 3
NONCOMPETITIVE_REL = 0.10
EXHAUST_MIN_VARIANTS = 2

tree = json.loads((AT / "experiment-tree.json").read_text())
exps = tree["experiments"]
global_best = min(e["cv_rmse"] for e in exps)

by_class = defaultdict(list)
for e in exps:
    by_class[e["arch_class"]].append(e["cv_rmse"])

tier1 = {}
for cls, scores in by_class.items():
    n = len(scores)
    best = min(scores)
    rel_gap = (best - global_best) / global_best
    exhausted = (n >= EXHAUST_MIN_VARIANTS) or (rel_gap > NONCOMPETITIVE_REL)
    reason = ("multiple_variants" if n >= EXHAUST_MIN_VARIANTS
              else f"non_competitive(+{rel_gap*100:.0f}%)")
    tier1[cls] = {"n_variants": n, "best": best, "exhausted": exhausted, "reason": reason}

n_classes = len(by_class)
tier2_ok = n_classes >= MIN_CLASSES
all_exhausted = all(v["exhausted"] for v in tier1.values())
converged = tier2_ok and all_exhausted

tree["convergence"] = {
    "tier1_within_class": tier1,
    "tier2_coverage": {"n_classes": n_classes, "min_required": MIN_CLASSES, "satisfied": tier2_ok},
    "global_best_cv_rmse": global_best,
}
tree["global_status"] = "CONVERGED" if converged else "EXPLORING"
(AT / "experiment-tree.json").write_text(json.dumps(tree, indent=2))

print("TIER 1 (within-class exhaustion):")
for cls, v in sorted(tier1.items()):
    print(f"  {cls:12} variants={v['n_variants']} best={v['best']:.5f} "
          f"exhausted={v['exhausted']} ({v['reason']})")
print(f"TIER 2 (coverage): {n_classes} classes >= {MIN_CLASSES} required -> {tier2_ok}")
print(f"global_best_cv_rmse={global_best:.5f}")
print(f"VERDICT: {tree['global_status']}")
sys.exit(0)

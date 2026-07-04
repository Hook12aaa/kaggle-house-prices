import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from sklearn.ensemble import GradientBoostingRegressor

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="tree_based",
    make_model=lambda: GradientBoostingRegressor(n_estimators=1500, learning_rate=0.02,
        max_depth=3, subsample=0.8, max_features="sqrt", random_state=0),
    needs_scaling=False,
    notes="sklearn GradientBoosting depth3 lr0.02 n1500",
)

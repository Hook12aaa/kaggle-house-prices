import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from lightgbm import LGBMRegressor

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="tree_based",
    make_model=lambda: LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.4, min_child_samples=10,
        reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1),
    needs_scaling=False,
    notes="LightGBM num_leaves15 lr0.02 n2000",
)

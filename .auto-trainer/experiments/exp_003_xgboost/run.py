import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from xgboost import XGBRegressor

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="tree_based",
    make_model=lambda: XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=3,
        subsample=0.7, colsample_bytree=0.4, min_child_weight=2, reg_lambda=1.0,
        reg_alpha=0.0, random_state=0, n_jobs=4),
    needs_scaling=False,
    notes="XGBoost depth3 lr0.02 n2000 subsample0.7 colsample0.4",
)

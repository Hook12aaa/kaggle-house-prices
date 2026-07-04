import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from sklearn.linear_model import Ridge

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="linear",
    make_model=lambda: Ridge(alpha=10.0, random_state=0),
    needs_scaling=True,
    notes="baseline: Ridge(alpha=10) on standardized design matrix",
)

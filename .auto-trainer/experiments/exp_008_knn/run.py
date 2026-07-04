import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from sklearn.neighbors import KNeighborsRegressor

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="neighbors",
    make_model=lambda: KNeighborsRegressor(n_neighbors=10, weights="distance", p=1),
    needs_scaling=True,
    notes="KNN k10 distance-weighted manhattan",
)

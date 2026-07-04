import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from sklearn.linear_model import ElasticNetCV
import numpy as np

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="linear",
    make_model=lambda: ElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,1.0], alphas=np.logspace(-4,-1,30), max_iter=5000, cv=5, random_state=0),
    needs_scaling=True,
    notes="ElasticNetCV grid over l1_ratio x alpha on standardized matrix",
)

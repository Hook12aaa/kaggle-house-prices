import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from sklearn.svm import SVR

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="svm",
    make_model=lambda: SVR(kernel="rbf", C=20.0, epsilon=0.01, gamma="scale"),
    needs_scaling=True,
    notes="SVR rbf C20 eps0.01 on standardized matrix",
)

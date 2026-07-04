import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from common import run_experiment
from sklearn.kernel_ridge import KernelRidge

run_experiment(
    Path(__file__).resolve().parent,
    arch_class="svm",
    make_model=lambda: KernelRidge(alpha=0.5, kernel="polynomial", degree=2, coef0=2.5),
    needs_scaling=True,
    notes="KernelRidge polynomial degree2 alpha0.5",
)

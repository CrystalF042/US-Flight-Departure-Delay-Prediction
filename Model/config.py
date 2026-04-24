from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
SAMPLE_SIZE = 2_000_000
N_FOLDS = 3

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_MODELING_DIR = REPORTS_DIR / "figures" / "modeling"


def ensure_project_dirs() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_MODELING_DIR.mkdir(parents=True, exist_ok=True)
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_baseline_lookups(
    baselines: dict[str, Any],
    output_path: str | Path = "models/baseline_lookups.pkl",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(baselines, f)

    return output_path
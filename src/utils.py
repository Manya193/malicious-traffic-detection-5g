from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
SELECTED_FEATURES_DIR = RESULTS_DIR / "selected_features"


def ensure_project_dirs() -> None:
    for path in [
        DATA_RAW,
        DATA_PROCESSED,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        SELECTED_FEATURES_DIR,
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "app",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_column_name(column: str) -> str:
    cleaned = str(column).strip()
    cleaned = " ".join(cleaned.replace("\ufeff", "").split())
    return cleaned


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

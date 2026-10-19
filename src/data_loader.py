from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.download_dataset import extract_archives, print_required_files
from src.utils import DATA_RAW, normalize_column_name, print_section


@dataclass
class LoadedDataset:
    data: pd.DataFrame
    files: list[Path]


class DatasetNotFoundError(FileNotFoundError):
    pass


def discover_csv_files(raw_dir: Path = DATA_RAW) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for archive in raw_dir.rglob("*.zip"):
        if zipfile.is_zipfile(archive):
            extract_archives(raw_dir)
            break
    return sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, low_memory=False, encoding="latin1")
    frame.columns = [normalize_column_name(col) for col in frame.columns]
    return frame


def find_label_column(columns: list[str]) -> str:
    normalized = {normalize_column_name(col).lower(): col for col in columns}
    for candidate in ("label", "class", "attack"):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("No label column found. CICIDS2017 CSV files normally contain a 'Label' column.")


def load_cicids2017(raw_dir: Path = DATA_RAW) -> LoadedDataset:
    print_section("Dataset Discovery")
    csv_files = discover_csv_files(raw_dir)
    print("Discovered CSV files:")
    if not csv_files:
        print("  - none")
        print_required_files(raw_dir)
        raise DatasetNotFoundError(f"No CICIDS2017 CSV files found under {raw_dir}")
    for path in csv_files:
        print(f"  - {path}")

    frames: list[pd.DataFrame] = []
    canonical_columns: list[str] | None = None
    skipped: list[Path] = []
    for path in csv_files:
        frame = _read_csv(path)
        if frame.empty:
            skipped.append(path)
            continue
        if canonical_columns is None:
            canonical_columns = list(frame.columns)
        common = [col for col in canonical_columns if col in frame.columns]
        if "Label" not in common and not any(col.lower() == "label" for col in common):
            skipped.append(path)
            continue
        frames.append(frame[common].copy())

    if not frames:
        raise DatasetNotFoundError("CSV files were found, but none contained compatible CICIDS2017 labeled flow columns.")

    data = pd.concat(frames, ignore_index=True)
    data.columns = [normalize_column_name(col) for col in data.columns]
    print(f"Loaded shape before cleaning: {data.shape}")
    print("Available columns:")
    print(", ".join(data.columns))
    if skipped:
        print("Skipped incompatible/empty files:")
        for path in skipped:
            print(f"  - {path}")
    return LoadedDataset(data=data, files=csv_files)

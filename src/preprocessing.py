from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import find_label_column
from src.utils import normalize_column_name, print_section


TARGET_COLUMN = "binary_label"
NORMAL_CLASS = "NORMAL"
MALICIOUS_CLASS = "MALICIOUS"
CLASS_ORDER = [NORMAL_CLASS, MALICIOUS_CLASS]


def _safe_print_mapping(mapping: dict) -> None:
    print(json.dumps(mapping, indent=2, ensure_ascii=True))


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_removed: int = 0
    empty_rows_removed: int = 0
    rows_removed_missing_label: int = 0
    infinity_values_replaced: int = 0
    missing_values_before: int = 0
    missing_values_after: int = 0
    original_label_distribution: dict[str, int] = field(default_factory=dict)
    binary_label_distribution: dict[str, int] = field(default_factory=dict)


IDENTIFIER_COLUMNS = {
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "timestamp",
    "time stamp",
    "simillarhttp",
}


def convert_labels(labels: pd.Series) -> pd.Series:
    clean = labels.astype(str).str.strip()
    return np.where(clean.str.upper() == "BENIGN", NORMAL_CLASS, MALICIOUS_CLASS)


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    print_section("Data Cleaning")
    frame = data.copy()
    frame.columns = [normalize_column_name(col) for col in frame.columns]
    rows_before, columns_before = frame.shape
    label_column = find_label_column(list(frame.columns))

    original_distribution = frame[label_column].astype(str).str.strip().value_counts(dropna=False).to_dict()
    print("Original label distribution:")
    _safe_print_mapping(original_distribution)

    empty_rows = int(frame.isna().all(axis=1).sum())
    frame = frame.dropna(how="all")

    missing_label = int(frame[label_column].isna().sum())
    frame = frame.dropna(subset=[label_column])

    duplicated = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()

    feature_probe = frame.drop(columns=[label_column], errors="ignore")
    numeric_probe = feature_probe.copy()
    for col in feature_probe.columns:
        if pd.api.types.is_numeric_dtype(feature_probe[col]):
            continue
        converted = pd.to_numeric(feature_probe[col], errors="coerce")
        non_missing = int(feature_probe[col].notna().sum())
        converted_non_missing = int(converted.notna().sum())
        if non_missing > 0 and converted_non_missing == non_missing:
            numeric_probe[col] = converted
    numeric_columns = numeric_probe.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(numeric_probe[numeric_columns]).sum().sum()) if len(numeric_columns) else 0
    if inf_count:
        numeric_probe[numeric_columns] = numeric_probe[numeric_columns].replace([np.inf, -np.inf], np.nan)
        for col in numeric_columns:
            frame[col] = numeric_probe[col]

    frame[TARGET_COLUMN] = convert_labels(frame[label_column])
    binary_distribution = frame[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict()
    frame = frame.drop(columns=[label_column])

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=int(frame.shape[0]),
        columns_before=columns_before,
        columns_after=int(frame.shape[1]),
        duplicates_removed=duplicated,
        empty_rows_removed=empty_rows,
        rows_removed_missing_label=missing_label,
        infinity_values_replaced=inf_count,
        missing_values_before=int(data.isna().sum().sum()),
        missing_values_after=int(frame.isna().sum().sum()),
        original_label_distribution={str(k): int(v) for k, v in original_distribution.items()},
        binary_label_distribution={str(k): int(v) for k, v in binary_distribution.items()},
    )

    print(f"Rows before cleaning: {report.rows_before}")
    print(f"Rows after cleaning: {report.rows_after}")
    print(f"Columns before cleaning: {report.columns_before}")
    print(f"Columns after cleaning: {report.columns_after}")
    print(f"Empty rows removed: {report.empty_rows_removed}")
    print(f"Rows removed for missing label: {report.rows_removed_missing_label}")
    print(f"Duplicate records removed: {report.duplicates_removed}")
    print(f"Infinity values replaced with missing values: {report.infinity_values_replaced}")
    print(f"Missing values before cleaning: {report.missing_values_before}")
    print(f"Missing values after cleaning: {report.missing_values_after}")
    print("Final binary distribution:")
    _safe_print_mapping(report.binary_label_distribution)
    return frame, report


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y = data[TARGET_COLUMN].copy()
    x = data.drop(columns=[TARGET_COLUMN]).copy()
    removed: list[str] = []
    for col in list(x.columns):
        if normalize_column_name(col).lower() in IDENTIFIER_COLUMNS:
            removed.append(f"{col}: identifier/time field removed to reduce memorization and leakage risk")
            x = x.drop(columns=[col])
    return x, y, removed


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = list(x_train.select_dtypes(include=[np.number]).columns)
    categorical_features = [col for col in x_train.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]

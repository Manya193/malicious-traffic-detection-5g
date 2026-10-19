from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils import FIGURES_DIR, SELECTED_FEATURES_DIR, print_section


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    dropped_correlated_features: list[str]
    importance_table: pd.DataFrame
    selection_sample_size: int
    selection_sample_class_distribution: dict[str, int]


def _stratified_training_sample(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a deterministic stratified sample without looking at the test set."""
    sample_size = min(int(sample_size), len(x_train))
    if sample_size <= 0:
        raise ValueError("selection_sample_size must be positive")
    if sample_size == len(x_train):
        return x_train, y_train

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=sample_size, random_state=random_state
    )
    sample_indices, _ = next(splitter.split(x_train, y_train))
    return x_train.iloc[sample_indices], y_train.iloc[sample_indices]


def save_feature_analysis(
    x_train_transformed: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 30,
    correlation_threshold: float = 0.98,
    random_state: int = 42,
    selection_sample_size: int = 150_000,
) -> FeatureSelectionResult:
    print_section("Feature Analysis and Selection")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Feature selection can be quadratic in columns and expensive in rows. A
    # fixed stratified sample keeps both analyses representative while making
    # the cost practical. It is taken from training data after the split;
    # x_test is never passed to this function.
    x_selection, y_selection = _stratified_training_sample(
        x_train_transformed,
        y_train,
        sample_size=selection_sample_size,
        random_state=random_state,
    )
    actual_sample_size = len(x_selection)
    sample_distribution = {
        str(label): int(count)
        for label, count in y_selection.value_counts().sort_index().items()
    }
    pd.Series(sample_distribution, name="records").rename_axis("class").to_csv(
        SELECTED_FEATURES_DIR / "selection_sample_class_distribution.csv"
    )

    stats = x_selection.describe().transpose()
    stats.to_csv(SELECTED_FEATURES_DIR / "numeric_feature_statistics.csv")

    corr = x_selection.corr(numeric_only=True).abs()
    corr.to_csv(SELECTED_FEATURES_DIR / "train_correlation_matrix.csv")
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False, cbar_kws={"shrink": 0.7})
    plt.title(f"Training-Sample Feature Correlation (n={actual_sample_size:,})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    reduced = x_selection.drop(columns=dropped, errors="ignore")

    selector = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    selector.fit(reduced, y_selection)
    importance_table = pd.DataFrame(
        {"feature": reduced.columns, "importance": selector.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_table.to_csv(SELECTED_FEATURES_DIR / "feature_importance_selection_stage.csv", index=False)

    positive = importance_table[importance_table["importance"] > 0]
    selected = positive.head(max_features)["feature"].tolist()
    if not selected:
        selected = importance_table.head(min(max_features, len(importance_table)))["feature"].tolist()

    pd.Series(selected, name="selected_feature").to_csv(
        SELECTED_FEATURES_DIR / "selected_features.csv", index=False
    )
    pd.Series(dropped, name="dropped_correlated_feature").to_csv(
        SELECTED_FEATURES_DIR / "dropped_correlated_features.csv", index=False
    )

    print(f"Highly correlated features dropped using training data only: {len(dropped)}")
    print(f"Feature-selection sample: {actual_sample_size:,} stratified training rows")
    print(f"Feature-selection sample class distribution: {sample_distribution}")
    print(f"Selected features: {len(selected)}")
    for feature in selected:
        print(f"  - {feature}")
    return FeatureSelectionResult(
        selected,
        dropped,
        importance_table,
        actual_sample_size,
        sample_distribution,
    )

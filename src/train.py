from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import DatasetNotFoundError, load_cicids2017
from src.evaluate import evaluate_model
from src.feature_selection import save_feature_analysis
from src.preprocessing import (
    CLASS_ORDER,
    build_preprocessor,
    clean_dataset,
    get_transformed_feature_names,
    split_features_target,
)
from src.utils import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT, ensure_project_dirs, print_section, write_json


def plot_class_distribution(y: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, order=CLASS_ORDER)
    plt.title("Binary Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()


def plot_feature_importance(importances: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = importances.head(25)
    plt.figure(figsize=(10, max(5, 0.28 * len(top))))
    sns.barplot(data=top, y="feature", x="importance")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close()


def train(
    raw_dir: Path,
    test_size: float,
    random_state: int,
    max_features: int,
    n_estimators: int,
    selection_sample_size: int,
) -> dict:
    ensure_project_dirs()
    loaded = load_cicids2017(raw_dir)
    cleaned, cleaning_report = clean_dataset(loaded.data)
    x, y, removed_features = split_features_target(cleaned)
    plot_class_distribution(y)

    print_section("Train/Test Split")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print("Training class distribution:")
    print(y_train.value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict())
    print("Testing class distribution:")
    print(y_test.value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict())
    print("Feature scaling: not used because Random Forest does not require feature scaling.")
    if removed_features:
        print("Removed features:")
        for reason in removed_features:
            print(f"  - {reason}")

    preprocessor = build_preprocessor(x_train)
    x_train_array = preprocessor.fit_transform(x_train)
    x_test_array = preprocessor.transform(x_test)
    transformed_features = get_transformed_feature_names(preprocessor)
    x_train_t = pd.DataFrame(x_train_array, columns=transformed_features, index=x_train.index)
    x_test_t = pd.DataFrame(x_test_array, columns=transformed_features, index=x_test.index)

    selection = save_feature_analysis(
        x_train_t,
        y_train,
        max_features=max_features,
        random_state=random_state,
        selection_sample_size=selection_sample_size,
    )
    x_train_selected = x_train_t[selection.selected_features]
    x_test_selected = x_test_t[selection.selected_features]

    print_section("Random Forest Training")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    start = time.perf_counter()
    model.fit(x_train_selected, y_train)
    training_seconds = time.perf_counter() - start
    print(f"Training time seconds: {training_seconds:.3f}")
    print(f"Number of trees: {n_estimators}")
    print(f"Number of selected features: {len(selection.selected_features)}")

    metrics = evaluate_model(model, x_test_selected, y_test)

    importance_table = pd.DataFrame(
        {"feature": selection.selected_features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_table.to_csv(METRICS_DIR / "final_feature_importance.csv", index=False)
    plot_feature_importance(importance_table)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    artifact = {
        "pipeline": pipeline,
        "selected_features": selection.selected_features,
        "all_transformed_features": transformed_features,
        "class_order": CLASS_ORDER,
        "removed_features": removed_features,
    }
    model_path = MODELS_DIR / "random_forest_cicids2017.joblib"
    joblib.dump(artifact, model_path)

    report = {
        "dataset_files": [str(path.relative_to(PROJECT_ROOT)) for path in loaded.files],
        "records_before_preprocessing": cleaning_report.rows_before,
        "records_after_preprocessing": cleaning_report.rows_after,
        "columns_after_preprocessing": cleaning_report.columns_after,
        "selected_feature_count": len(selection.selected_features),
        "selected_features": selection.selected_features,
        "feature_selection": {
            "sample_size": selection.selection_sample_size,
            "sample_source": "training split only",
            "sample_class_distribution": selection.selection_sample_class_distribution,
            "correlation_threshold": 0.98,
        },
        "training_samples": int(len(x_train)),
        "testing_samples": int(len(x_test)),
        "random_forest": {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        "training_time_seconds": training_seconds,
        "metrics": metrics,
        "top_features": importance_table.head(15).to_dict(orient="records"),
        "cleaning_report": cleaning_report.__dict__,
        "model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "removed_features": removed_features,
    }
    write_json(METRICS_DIR / "training_summary.json", report)
    print_section("Evaluation")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision ({metrics['positive_class']}): {metrics['precision']:.6f}")
    print(f"Recall ({metrics['positive_class']}): {metrics['recall']:.6f}")
    print(f"F1-score ({metrics['positive_class']}): {metrics['f1_score']:.6f}")
    print(f"Confusion matrix {CLASS_ORDER}: {metrics['confusion_matrix']}")
    print(f"Saved model: {model_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Random Forest malicious traffic detector on CICIDS2017.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--selection-sample-size",
        type=int,
        default=150_000,
        help="Stratified training-only rows used for correlation and selection RF (default: 150000).",
    )
    args = parser.parse_args()
    try:
        train(
            args.raw_dir,
            args.test_size,
            args.random_state,
            args.max_features,
            args.n_estimators,
            args.selection_sample_size,
        )
    except DatasetNotFoundError as exc:
        print(f"Training stopped: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

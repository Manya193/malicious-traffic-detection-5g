from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"),
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.data_loader import DatasetNotFoundError, find_label_column, load_cicids2017
from src.preprocessing import MALICIOUS_CLASS, TARGET_COLUMN, clean_dataset, split_features_target
from src.utils import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT, ensure_project_dirs, print_section, write_json


ATTACK_LABEL_COLUMN = "_original_attack_label"
DEFAULT_BINARY_MODEL = MODELS_DIR / "random_forest_cicids2017.joblib"
DEFAULT_ATTACK_MODEL = MODELS_DIR / "random_forest_attack_type.joblib"


def _load_malicious_records(raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[Path], dict]:
    loaded = load_cicids2017(raw_dir)
    label_column = find_label_column(list(loaded.data.columns))
    source = loaded.data.copy()
    # Preserve the original CICIDS2017 category before clean_dataset performs
    # the required BENIGN/NORMAL versus attack/MALICIOUS conversion.
    source[ATTACK_LABEL_COLUMN] = source[label_column].astype(str).str.strip()
    cleaned, cleaning_report = clean_dataset(source)
    malicious = cleaned[cleaned[TARGET_COLUMN] == MALICIOUS_CLASS].copy()
    labels = malicious[ATTACK_LABEL_COLUMN].astype(str).str.strip()
    feature_source = malicious.drop(columns=[ATTACK_LABEL_COLUMN])
    x, _, removed_features = split_features_target(feature_source)
    labels = labels.loc[x.index]
    return x, labels, loaded.files, {
        "cleaning_report": cleaning_report.__dict__,
        "removed_features": removed_features,
    }


def _plot_class_distribution(labels: pd.Series) -> None:
    counts = labels.value_counts().sort_values(ascending=True)
    plt.figure(figsize=(10, max(5, 0.34 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index, color="#2f6f9f")
    plt.title("CICIDS2017 Malicious Attack-Type Distribution")
    plt.xlabel("Records")
    plt.ylabel("Original CICIDS2017 Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attack_type_class_distribution.png", dpi=160)
    plt.close()


def _plot_confusion_matrix(matrix: list[list[int]], classes: list[str]) -> None:
    width = max(8, 0.65 * len(classes))
    plt.figure(figsize=(width, width * 0.82))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted original attack label")
    plt.ylabel("Actual original attack label")
    plt.title("Attack-Type Random Forest Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attack_type_confusion_matrix.png", dpi=160)
    plt.close()


def train_attack_type(
    raw_dir: Path,
    binary_model_path: Path,
    attack_model_path: Path,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> dict:
    ensure_project_dirs()
    if not binary_model_path.exists():
        raise FileNotFoundError(f"Existing binary model not found: {binary_model_path}")
    if attack_model_path.resolve() == binary_model_path.resolve():
        raise ValueError("Attack-type model path must be different from the binary model path")

    print_section("Attack-Type Dataset Preparation")
    binary_artifact = joblib.load(binary_model_path)
    preprocessor = binary_artifact["pipeline"].named_steps["preprocessor"]
    selected_features = list(binary_artifact["selected_features"])
    transformed_features = list(binary_artifact["all_transformed_features"])

    x, labels, files, preparation = _load_malicious_records(raw_dir)
    classes = sorted(labels.unique().tolist())
    if len(classes) < 2:
        raise ValueError("At least two original attack labels are required")
    print(f"Malicious records: {len(x):,}")
    print(f"Attack classes: {len(classes)}")
    for label, count in labels.value_counts().sort_index().items():
        print(f"  {label}: {count:,}")
    _plot_class_distribution(labels)
    labels.value_counts().sort_index().rename("records").to_csv(
        METRICS_DIR / "attack_type_class_distribution.csv", header=True
    )

    print_section("Attack-Type Train/Test Split")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    print(f"Training samples: {len(x_train):,}")
    print(f"Testing samples: {len(x_test):,}")

    x_train_transformed = pd.DataFrame(
        preprocessor.transform(x_train),
        columns=transformed_features,
        index=x_train.index,
    )[selected_features]
    x_test_transformed = pd.DataFrame(
        preprocessor.transform(x_test),
        columns=transformed_features,
        index=x_test.index,
    )[selected_features]

    print_section("Attack-Type Random Forest Training")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    start = time.perf_counter()
    model.fit(x_train_transformed, y_train)
    training_seconds = time.perf_counter() - start
    print(f"Training time seconds: {training_seconds:.3f}")

    y_pred = model.predict(x_test_transformed)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=classes,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "class_order": classes,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "averaging": "weighted",
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_json(METRICS_DIR / "attack_type_evaluation_metrics.json", metrics)
    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR / "attack_type_classification_report.csv"
    )
    _plot_confusion_matrix(matrix.tolist(), classes)

    importance = pd.DataFrame(
        {"feature": selected_features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(METRICS_DIR / "attack_type_feature_importance.csv", index=False)

    attack_artifact = {
        "classifier": model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "all_transformed_features": transformed_features,
        "class_order": classes,
        "source_label_column": "Label",
        "binary_model_file": str(binary_model_path.relative_to(PROJECT_ROOT)),
    }
    joblib.dump(attack_artifact, attack_model_path)

    summary = {
        "dataset_files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "source_label_column": "Label",
        "attack_classes": classes,
        "attack_class_count": len(classes),
        "class_distribution": {
            str(label): int(count) for label, count in labels.value_counts().sort_index().items()
        },
        "training_class_distribution": {
            str(label): int(count) for label, count in y_train.value_counts().sort_index().items()
        },
        "testing_class_distribution": {
            str(label): int(count) for label, count in y_test.value_counts().sort_index().items()
        },
        "training_samples": int(len(x_train)),
        "testing_samples": int(len(x_test)),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "random_forest": {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        "training_time_seconds": training_seconds,
        "metrics": metrics,
        "model_file": str(attack_model_path.relative_to(PROJECT_ROOT)),
        "binary_model_file": str(binary_model_path.relative_to(PROJECT_ROOT)),
        **preparation,
    }
    write_json(METRICS_DIR / "attack_type_training_summary.json", summary)

    print_section("Attack-Type Evaluation")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Weighted precision: {precision:.6f}")
    print(f"Weighted recall: {recall:.6f}")
    print(f"Weighted F1-score: {f1:.6f}")
    print(f"Confusion matrix: {METRICS_DIR / 'attack_type_evaluation_metrics.json'}")
    print(f"Saved model: {attack_model_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the second-stage CICIDS2017 attack-type Random Forest."
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--binary-model", type=Path, default=DEFAULT_BINARY_MODEL)
    parser.add_argument("--attack-model", type=Path, default=DEFAULT_ATTACK_MODEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    try:
        train_attack_type(
            args.raw_dir,
            args.binary_model,
            args.attack_model,
            args.test_size,
            args.random_state,
            args.n_estimators,
        )
    except (DatasetNotFoundError, FileNotFoundError, ValueError) as exc:
        print(f"Attack-type training stopped: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

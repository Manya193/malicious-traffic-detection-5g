from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.preprocessing import CLASS_ORDER, MALICIOUS_CLASS
from src.utils import FIGURES_DIR, METRICS_DIR, write_json


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[MALICIOUS_CLASS], average="binary", pos_label=MALICIOUS_CLASS, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True)

    metrics = {
        "positive_class": MALICIOUS_CLASS,
        "class_order": CLASS_ORDER,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_json(METRICS_DIR / "evaluation_metrics.json", metrics)
    pd.DataFrame(report).transpose().to_csv(METRICS_DIR / "classification_report.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    return metrics

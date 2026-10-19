from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"),
)

import joblib
import numpy as np
import pandas as pd
import shap

from src.predict import ATTACK_TYPE_MODEL
from src.utils import MODELS_DIR, normalize_column_name


def _class_shap_values(
    shap_values: object,
    expected_value: object,
    class_index: int,
    feature_count: int,
) -> tuple[np.ndarray, float]:
    """Normalize SHAP's multiclass return formats across supported versions."""
    values = np.asarray(shap_values)
    expected = np.asarray(expected_value).reshape(-1)

    if isinstance(shap_values, list):
        contributions = np.asarray(shap_values[class_index])[0]
    elif values.ndim == 3:
        # SHAP >= 0.45 returns (rows, features, classes) for this model.
        contributions = values[0, :, class_index]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}")

    if len(contributions) != feature_count:
        raise ValueError(
            f"SHAP returned {len(contributions)} contributions for {feature_count} features"
        )
    base_value = float(expected[class_index]) if len(expected) > 1 else float(expected[0])
    return np.asarray(contributions, dtype=float), base_value


def explain_attack_type_record(
    records: pd.DataFrame,
    row_position: int,
    model_path: Path = ATTACK_TYPE_MODEL,
) -> dict:
    """Create a local SHAP explanation for one already-uploaded traffic row.

    This function deliberately explains only the second-stage classifier. It
    reuses that artifact's fitted preprocessor and selected features and does
    not call or alter the binary prediction pipeline.
    """
    if row_position < 0 or row_position >= len(records):
        raise IndexError("The requested traffic row is outside the uploaded data")
    if not model_path.exists():
        raise FileNotFoundError(f"Attack-type model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    preprocessor = artifact["preprocessor"]
    selected_features = list(artifact["selected_features"])
    transformed_names = list(artifact["all_transformed_features"])
    classifier = artifact["classifier"]

    normalized_records = records.copy()
    normalized_records.columns = [
        normalize_column_name(column) for column in normalized_records.columns
    ]
    transformed = preprocessor.transform(normalized_records)
    transformed_frame = pd.DataFrame(
        transformed,
        columns=transformed_names,
        index=normalized_records.index,
    )[selected_features]
    row = transformed_frame.iloc[[row_position]]
    predicted_class = str(classifier.predict(row)[0])
    class_index = list(classifier.classes_).index(predicted_class)

    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(row, check_additivity=False)
    contributions, base_value = _class_shap_values(
        raw_values,
        explainer.expected_value,
        class_index,
        len(selected_features),
    )

    original_row = normalized_records.iloc[row_position]
    explanation = pd.DataFrame(
        {
            "Feature": selected_features,
            "Actual Value": [original_row.get(feature, np.nan) for feature in selected_features],
            "SHAP Contribution": contributions,
        }
    )
    explanation["Effect"] = np.where(
        explanation["SHAP Contribution"] >= 0,
        "Supports prediction",
        "Opposes prediction",
    )
    explanation["Absolute Contribution"] = explanation["SHAP Contribution"].abs()
    explanation = explanation.sort_values(
        "Absolute Contribution", ascending=False
    ).drop(columns=["Absolute Contribution"])
    explanation = explanation.reset_index(drop=True)

    return {
        "predicted_attack_type": predicted_class,
        "base_value": base_value,
        "feature_contributions": explanation,
    }


__all__ = ["explain_attack_type_record"]

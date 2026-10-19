from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import MALICIOUS_CLASS, NORMAL_CLASS
from src.utils import MODELS_DIR, normalize_column_name

ATTACK_TYPE_MODEL = MODELS_DIR / "random_forest_attack_type.joblib"


def load_artifact(model_path: Path = MODELS_DIR / "random_forest_cicids2017.joblib") -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Train the model first.")
    return joblib.load(model_path)


def load_attack_type_artifact(model_path: Path = ATTACK_TYPE_MODEL) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Attack-type model artifact not found: {model_path}")
    return joblib.load(model_path)


def predict_records(
    records: pd.DataFrame,
    model_path: Path = MODELS_DIR / "random_forest_cicids2017.joblib",
    attack_model_path: Path = ATTACK_TYPE_MODEL,
) -> pd.DataFrame:
    # CICIDS2017 CSV headers commonly contain leading spaces. Match the
    # normalization applied by the training loader before transforming input.
    records = records.copy()
    records.columns = [normalize_column_name(column) for column in records.columns]
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    selected = artifact["selected_features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(records)
    feature_names = artifact["all_transformed_features"]
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=records.index)
    predictions = classifier.predict(transformed_frame[selected])
    output = pd.DataFrame({"prediction": predictions}, index=records.index)
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(transformed_frame[selected])
        for idx, class_name in enumerate(classifier.classes_):
            output[f"probability_{class_name}"] = probabilities[:, idx]
        output["confidence"] = probabilities.max(axis=1)
    attack_types = pd.Series("", index=records.index, dtype="object")
    if attack_model_path.exists() and (output["prediction"] == MALICIOUS_CLASS).any():
        attack_artifact = load_attack_type_artifact(attack_model_path)
        attack_classifier = attack_artifact["classifier"]
        attack_selected = attack_artifact["selected_features"]
        attack_types.loc[output["prediction"] == MALICIOUS_CLASS] = attack_classifier.predict(
            transformed_frame.loc[output["prediction"] == MALICIOUS_CLASS, attack_selected]
        )
    output["attack_type"] = attack_types
    output["message"] = output["prediction"].map(
        {
            MALICIOUS_CLASS: "ALERT: MALICIOUS TRAFFIC DETECTED",
            NORMAL_CLASS: "NORMAL: SAFE TRAFFIC",
        }
    )
    malicious_mask = output["prediction"] == MALICIOUS_CLASS
    output.loc[malicious_mask, "message"] = (
        "MALICIOUS: Attack Type: " + output.loc[malicious_mask, "attack_type"].astype(str)
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict NORMAL/MALICIOUS for new traffic CSV records.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "random_forest_cicids2017.joblib")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = pd.read_csv(args.input_csv)
    predictions = predict_records(records, args.model)
    print(predictions.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

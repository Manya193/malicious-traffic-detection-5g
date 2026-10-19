from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import (
    ATTACK_TYPE_MODEL,
    load_artifact,
    load_attack_type_artifact,
    predict_records,
)
from src.utils import METRICS_DIR, MODELS_DIR
from src.xai import explain_attack_type_record


st.set_page_config(
    page_title="5G Network Traffic Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ 5G Network Traffic Detector")
st.caption("Machine-learning-based malicious traffic detection")


model_path = MODELS_DIR / "random_forest_cicids2017.joblib"
summary_path = METRICS_DIR / "training_summary.json"


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Model Information")

    if model_path.exists():
        artifact = load_artifact(model_path)
        classifier = artifact["pipeline"].named_steps["classifier"]

        # Model status
        st.success("Model loaded")

        # Model details
        st.write(f"**Classifier:** Random Forest")
        st.write(f"**Trees:** {classifier.n_estimators:,}")
        st.write(
            f"**Selected features:** "
            f"{len(artifact['selected_features'])}"
        )

        # -----------------------------------------------------------
        # MODEL PERFORMANCE
        # -----------------------------------------------------------

        if summary_path.exists():
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

            metrics = summary.get("metrics", {})

            st.divider()
            st.subheader("Model Performance")

            # Convert decimal scores to percentages
            accuracy = metrics.get("accuracy", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1_score", 0)

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

            st.metric(
                "F1-score",
                f"{f1_score * 100:.2f}%"
            )

            # -------------------------------------------------------
            # DATASET INFORMATION
            # -------------------------------------------------------

            st.divider()
            st.caption("Dataset")

            st.write(
                f"**Training rows:** "
                f"{summary.get('training_samples', 0):,}"
            )

            st.write(
                f"**Test rows:** "
                f"{summary.get('testing_samples', 0):,}"
            )

    else:
        st.error("Saved model not found")

# -------------------------------------------------------------------
# TRAFFIC INPUT
# -------------------------------------------------------------------

st.header("Traffic Analysis")

st.write(
    "Upload a CSV file to analyze network traffic."
)

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    label_visibility="collapsed",
)


# -------------------------------------------------------------------
# MODEL AVAILABILITY CHECK
# -------------------------------------------------------------------

if not model_path.exists():

    st.warning(
        "The saved Random Forest model is unavailable. "
        "Run the training pipeline first."
    )


# -------------------------------------------------------------------
# CSV UPLOAD AND VALIDATION
# -------------------------------------------------------------------

elif uploaded is not None:

    # Try reading the uploaded CSV
    try:
        records = pd.read_csv(uploaded)

    except Exception as exc:
        st.error(
            "❌ Unable to read this CSV file. "
            "Please check that it is a valid CSV file."
        )

        st.caption(f"Technical details: {exc}")
        st.stop()

    # Check for empty CSV
    if records.empty:
        st.warning(
            "⚠️ The uploaded CSV is empty. "
            "Please upload a CSV containing network traffic records."
        )
        st.stop()

    # Load the trained model artifact
    try:
        artifact = load_artifact(model_path)
        required_features = artifact["selected_features"]

    except Exception as exc:
        st.error(
            "❌ The trained model could not be loaded."
        )
        st.caption(f"Technical details: {exc}")
        st.stop()

    # ---------------------------------------------------------------
    # Validate required features
    # ---------------------------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in records.columns
    ]

    if missing_features:

        st.error(
            "❌ Invalid traffic CSV"
        )

        st.write(
            "The uploaded CSV does not contain the required "
            "network-traffic features."
        )

        st.warning(
            f"Missing {len(missing_features)} required feature(s)."
        )

        with st.expander("View missing features"):
            for feature in missing_features:
                st.write(f"- `{feature}`")

        st.info(
            "Please upload a CSV containing the network traffic "
            "features used by the trained model."
        )

        st.stop()

    # ---------------------------------------------------------------
    # CSV is valid
    # ---------------------------------------------------------------

    st.success(
        f"✅ CSV validated successfully — "
        f"{len(records):,} traffic record(s) loaded."
    )

    with st.expander("View traffic input", expanded=True):
        st.dataframe(
            records.head(25),
            width="stretch"
        )

    # ----------------------------------------------------------------
    # RUN PREDICTION
    # ----------------------------------------------------------------

    if st.button(
        "Run prediction",
        type="primary",
        icon="🔍"
    ):

        try:

            predictions = predict_records(
                records,
                model_path
            )

        except Exception as exc:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.caption(
                f"Technical details: {exc}"
            )

        else:

            st.session_state["xai_records"] = records.copy()
            st.session_state["xai_predictions"] = predictions.copy()
            st.session_state["xai_upload_name"] = uploaded.name

    predictions = st.session_state.get("xai_predictions")
    xai_records = st.session_state.get("xai_records")
    if predictions is not None and xai_records is not None:

            malicious_count = int(
                (
                    predictions["prediction"] == "MALICIOUS"
                ).sum()
            )

            normal_count = int(
                (
                    predictions["prediction"] == "NORMAL"
                ).sum()
            )

            # --------------------------------------------------------
            # Prediction Result
            # --------------------------------------------------------

            st.subheader("Prediction Result")

            result_col, normal_col, malicious_col = st.columns(3)

            with result_col:

                if malicious_count:

                    st.error(
                        "🔴 MALICIOUS"
                    )

                else:

                    st.success(
                        "🟢 NORMAL"
                    )

            normal_col.metric(
                "NORMAL",
                f"{normal_count:,}"
            )

            malicious_col.metric(
                "MALICIOUS",
                f"{malicious_count:,}"
            )

            # --------------------------------------------------------
            # Attack Type
            # --------------------------------------------------------

            if malicious_count:

                attack_types = sorted(
                    predictions.loc[
                        predictions["prediction"] == "MALICIOUS",
                        "attack_type"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

                st.subheader(
                    "Detected Attack Type"
                )

                for attack_type in attack_types:

                    st.warning(
                        f"⚠️ Attack Type: {attack_type}"
                    )

            # --------------------------------------------------------
            # Detailed predictions
            # --------------------------------------------------------

            st.subheader(
                "Detailed Prediction Results"
            )

            detailed_predictions = predictions.copy()
            detailed_predictions["XAI Explanation"] = detailed_predictions[
                "prediction"
            ].map(
                {
                    "MALICIOUS": "🔍 Explain",
                    "NORMAL": "—",
                }
            )
            st.dataframe(
                detailed_predictions,
                width="stretch"
            )

            malicious_positions = [
                int(position)
                for position, value in enumerate(predictions["prediction"])
                if value == "MALICIOUS"
            ]
            if malicious_positions:
                st.subheader("Per-record XAI")
                selected_position = st.selectbox(
                    "Select a malicious row marked 🔍 Explain",
                    options=malicious_positions,
                    format_func=lambda position: (
                        f"Row {position + 1} — "
                        f"{predictions.iloc[position]['attack_type']}"
                    ),
                )
                if st.button(
                    "🔍 Explain selected row",
                    key="explain_selected_attack_row",
                ):
                    try:
                        local_explanation = explain_attack_type_record(
                            xai_records,
                            selected_position,
                        )
                    except Exception as exc:
                        st.error("The selected row could not be explained.")
                        st.caption(f"Technical details: {exc}")
                    else:
                        predicted_attack = local_explanation["predicted_attack_type"]
                        st.markdown(
                            f"### Why was this traffic classified as {predicted_attack}?"
                        )
                        st.info(
                            "The attack-type model classified this flow based on the "
                            "combined contribution of the network-flow features shown "
                            "below. Positive contributions support the predicted class, "
                            "while negative contributions oppose it. No individual "
                            "feature should be interpreted as the sole cause."
                        )
                        st.dataframe(
                            local_explanation["feature_contributions"],
                            width="stretch",
                            hide_index=True,
                        )


# -------------------------------------------------------------------
# NO FILE UPLOADED
# -------------------------------------------------------------------

else:

    st.info(
        "Upload a CSV file to run a traffic prediction."
    )

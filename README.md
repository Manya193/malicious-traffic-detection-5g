# Malicious Traffic Detection in 5G Using Machine Learning

Two-stage malicious network traffic detection using the official CICIDS2017 dataset and `RandomForestClassifier` models.

## Project Facts

- Dataset: CICIDS2017
- Official source: Canadian Institute for Cybersecurity, University of New Brunswick
- Classification: Two-stage binary plus attack-type identification
- Classes: `NORMAL`, `MALICIOUS`
- Label conversion: `BENIGN` -> `NORMAL`; all other labels -> `MALICIOUS`
- Models: Random Forest only

## Dataset Preparation

Use the official CIC/UNB dataset page:

https://www.unb.ca/cic/datasets/ids-2017.html

The official page states that `MachineLearningCSV.zip` and `GeneratedLabelledFlows.zip` are available for machine learning and labeled-flow research use through the CIC dataset download form.

Place one of these official archives, or the extracted CSV files, under:

```text
data/raw/
```

Preferred file:

```text
data/raw/MachineLearningCSV.zip
```

Acceptable alternative:

```text
data/raw/GeneratedLabelledFlows.zip
```

The project never creates fake CICIDS2017 records. If no real CSV files are present, training stops with a clear dataset-required message.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Optional Official Download Attempt

```bash
python -m src.download_dataset
```

If the CIC endpoint returns the official ZIP, it is saved to `data/raw/` and extracted. If the endpoint returns the CIC form/error HTML instead, manually download the dataset from the official page above and place the ZIP or extracted CSV files in `data/raw/`.

## Train

```bash
python -m src.train
```

The training pipeline:

1. Discovers CSV files recursively under `data/raw/`.
2. Extracts ZIP archives if present.
3. Loads compatible CICIDS2017 CSV files.
4. Cleans missing, duplicate, empty, invalid, and infinity values.
5. Converts labels to `NORMAL` and `MALICIOUS`.
6. Splits train/test with stratification.
7. Fits preprocessing on training data only.
8. Performs correlation and Random Forest importance feature selection using training data only.
9. Trains the final Random Forest.
10. Saves metrics, plots, selected features, and model artifacts.

## Evaluate Results

Outputs are saved under:

```text
results/metrics/
results/figures/
results/selected_features/
models/random_forest_cicids2017.joblib
```

Important files:

- `results/metrics/evaluation_metrics.json`
- `results/metrics/training_summary.json`
- `results/metrics/classification_report.csv`
- `results/metrics/final_feature_importance.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/feature_importance.png`
- `results/selected_features/selected_features.csv`

Metrics are generated only by actual execution on the real dataset.

## Attack-Type Identification

After the existing binary model has been trained, train the separate second-stage classifier:

```bash
python -m src.train_attack_type
```

This command preserves the original CICIDS2017 `Label` values, keeps only records that became `MALICIOUS`, and trains a second stratified Random Forest on those original attack labels. It reuses the already fitted binary preprocessing transformer and the same 30 selected features. It does not retrain or overwrite the binary model.

The second-stage artifacts are separate:

- `models/random_forest_attack_type.joblib`
- `results/metrics/attack_type_training_summary.json`
- `results/metrics/attack_type_evaluation_metrics.json`
- `results/metrics/attack_type_classification_report.csv`
- `results/metrics/attack_type_class_distribution.csv`
- `results/metrics/attack_type_feature_importance.csv`
- `results/figures/attack_type_confusion_matrix.png`
- `results/figures/attack_type_class_distribution.png`

Prediction is two-stage: the existing binary model first returns `NORMAL` or `MALICIOUS`. Only malicious records are sent to the attack-type Random Forest, which returns an original CICIDS2017 attack label such as `DoS Hulk`, `PortScan`, or `DDoS`. Normal records do not receive an attack type.

## Predict New Traffic

```bash
python -m src.predict path\to\new_traffic.csv
```

Output is `NORMAL: SAFE TRAFFIC` or `MALICIOUS: Attack Type: <original CICIDS2017 label>` when the second-stage model is available.

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Upload a CSV containing the same raw feature columns used during training.

## Architecture

```text
data/
  raw/
  processed/
src/
  data_loader.py
  download_dataset.py
  preprocessing.py
  feature_selection.py
  train_attack_type.py
  train.py
  evaluate.py
  predict.py
  utils.py
app/
  streamlit_app.py
models/
results/
  figures/
  metrics/
  selected_features/
```

## Leakage Prevention

- The original label column is removed from input features.
- Identifier/time fields such as IP addresses, flow IDs, and timestamps are removed when present.
- Train/test split happens before learned preprocessing and feature selection.
- Imputation and encoding are fitted using training data only.
- Correlation filtering and the feature-importance selector use a deterministic stratified sample of 150,000 rows from the training split only. This keeps selection practical on CICIDS2017 while preserving the binary class balance; the final Random Forest is still fitted on all available training rows and evaluated on all held-out test rows. Use `--selection-sample-size` to choose another manageable training-only sample.
- Test data is used only for final evaluation.
- Original attack labels are preserved separately for malicious-only attack-type training; the binary model remains unchanged.

## Limitations

- The project requires real CICIDS2017 CSV data; no metrics are available until the official dataset is supplied.
- CICIDS2017 is not a native 5G dataset, but it is a standard IDS benchmark suitable for an academic malicious traffic detection pipeline.
- Random Forest is the only implemented classifier by design.

## Future Improvements

- Add controlled hyperparameter tuning for Random Forest.
- Add model card documentation after real training.
- Validate on newer 5G or IoT network intrusion datasets while keeping CICIDS2017 as the baseline.

# Malicious Traffic Detection in 5G Using Machine Learning

A machine learning-based network intrusion detection project that uses the CICIDS2017 dataset and Random Forest classifiers to identify normal and malicious network traffic, classify detected attacks, and explain predictions through SHAP-based explainable AI.

## Project Overview

The increasing complexity of network traffic creates a need for intelligent intrusion detection systems that can identify potentially malicious activities. This project develops a machine learning pipeline for traffic classification, attack identification, and prediction explainability.

The system uses a **two-stage Random Forest classification approach**:

1. **Binary Classification:** Classifies network traffic as `NORMAL` or `MALICIOUS`.
2. **Attack-Type Classification:** If traffic is classified as malicious, a second Random Forest model predicts its attack category.
3. **Explainable AI:** Uses SHAP-based explanations to help users understand the features contributing to a prediction.
4. **Interactive Web Application:** Provides a Streamlit interface for uploading traffic data and viewing predictions.


## Objectives

- Detect malicious network traffic using machine learning.
- Distinguish normal traffic from malicious traffic.
- Identify the attack category of traffic classified as malicious.
- Select relevant features using correlation analysis and Random Forest feature importance.
- Evaluate the models using standard classification metrics.
- Provide interpretable predictions through SHAP-based explainability.
- Develop an interactive application for testing traffic records.

## Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Dataset | CICIDS2017 |
| Machine learning | Scikit-learn |
| Classification algorithm | Random Forest |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainable AI | SHAP |
| Web application | Streamlit |
| Model serialization | Joblib |

## Dataset

The project uses the **CICIDS2017 dataset**, published by the Canadian Institute for Cybersecurity at the University of New Brunswick.

Dataset source: https://www.unb.ca/cic/datasets/ids-2017.html

CICIDS2017 contains labeled network traffic representing benign activity and different types of attacks. Its flow-based features support the development and evaluation of intrusion detection models.

### Label Processing

The original dataset labels are mapped as follows for binary classification:

- `BENIGN` → `NORMAL`
- All other attack labels → `MALICIOUS`

The original attack labels are retained separately for the second-stage attack-type classifier.

### Dataset Handling

The dataset must be obtained from the official source and placed under `data/raw/` before training.

The raw dataset files are excluded from GitHub to avoid uploading large datasets. The trained model files are included in the repository.

## System Architecture

```text
          CICIDS2017 Dataset
                   |
                   v
          Data Preprocessing
                   |
                   v
          Feature Selection
       (Correlation + RF Importance)
                   |
                   v
          Binary Random Forest
                   |
             +-----+-----+
             |           |
             v           v
          NORMAL      MALICIOUS
             |           |
             v           v
         Safe Traffic   Attack-Type
                        Random Forest
                            |
                            v
                    Predicted Attack
                            |
                            v
                    SHAP Explanation
                            |
                            v
                   Streamlit Interface
```

## Methodology

### 1. Data Preprocessing

The preprocessing stage prepares the dataset for model training by handling missing or invalid values, duplicate records, and infinite values where applicable. Labels are standardized, and unsuitable identifier or time-related fields are removed when present.

The dataset is divided into training and testing sets using stratified splitting to preserve class proportions.

### 2. Feature Selection

Feature selection combines two techniques:

- **Correlation analysis:** Identifies highly correlated features to reduce redundancy.
- **Random Forest feature importance:** Ranks features according to their contribution to classification.

A stratified sample of 150,000 records from the training split is used for feature selection to reduce computation time. The final binary Random Forest model is trained on the full training split using the selected 30 features.

### 3. Binary Classification

The first Random Forest model predicts whether an input traffic record is:

- `NORMAL` — classified as benign traffic.
- `MALICIOUS` — classified as potentially malicious traffic.

### 4. Attack-Type Classification

Traffic classified as malicious is passed to a separate Random Forest classifier trained on the original attack labels. The model predicts an attack category represented in the training data.

Normal traffic does not receive an attack-type prediction.

### 5. Explainable AI

SHAP-based explanations help interpret individual predictions by showing how input features contribute to the model output.

These explanations support model interpretation; they do not independently establish that a traffic record is malicious or prove the cause of an attack.

### 6. Web Application

The Streamlit application allows users to upload traffic data, view classification results, identify predicted attack categories when applicable, and inspect feature-level explanations.

Input data must contain the feature columns expected by the trained pipeline.

## Model Performance

The following results were obtained from the project's recorded held-out test evaluations.

### Binary Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 99.8708% |
| Precision | 99.6347% |
| Recall | 99.5996% |
| F1-score | 99.6172% |
| Selected features | 30 |
| Number of trees | 100 |

### Attack-Type Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 99.7981% |
| Weighted precision | 99.7853% |
| Weighted recall | 99.7981% |
| Weighted F1-score | 99.7900% |
| Number of classes | 14 |

*These are experimental results on CICIDS2017. Performance on different network environments or real 5G traffic may differ.*

## Project Structure

```text
Malicious Traffic Detection/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── random_forest_cicids2017.joblib
│   └── random_forest_attack_type.joblib
├── notebooks/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── selected_features/
├── src/
│   ├── data_loader.py
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── feature_selection.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_attack_type.py
│   ├── utils.py
│   └── xai.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

### Prerequisites

- Python installed on your system.
- Git, if cloning the repository.
- The official CICIDS2017 dataset for retraining the models.

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/malicious-traffic-detection-5g.git
cd malicious-traffic-detection-5g
```

Replace `USERNAME` with the GitHub account that hosts the repository.

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If the virtual environment is already activated, you can skip this step.

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application from the project root:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

Upload a CSV file containing the expected input feature columns to test the classifier.

## Deployment and Hosting Limitations

The application is deployed on Render using its **Free plan** for demonstration purposes. The free hosting environment has limited available memory (512 MB in the instance observed during testing), which can cause the application to exceed its resource limit while loading the trained Random Forest models or executing prediction and SHAP-based explainability operations.

As a result, the online demo may not support every prediction workload reliably. This limitation is related to the available hosting resources and does not, by itself, indicate a problem with the trained models.

**Live Demo:** https://malicious-traffic-detection-5g.onrender.com

### Recommended Testing Method

For complete testing of binary classification, attack-type identification, and SHAP-based explanations, running the application locally is recommended.

1. Clone the GitHub repository.
2. Install the dependencies using `requirements.txt`.
3. Ensure both trained model files are present in the `models/` directory.
4. Start the application:

```bash
streamlit run app/streamlit_app.py
```

5. Open the local URL provided in the terminal and upload a compatible traffic CSV file.

Local execution avoids Render's specific 512 MB hosting limit.

The online deployment is provided as a demonstration endpoint, while local execution is recommended for reliable functional testing.

## Training the Models

Training requires the official CICIDS2017 dataset under `data/raw/`.

### Train the Binary Classifier

```bash
python -m src.train
```

This runs the binary training pipeline and generates the associated model and evaluation artifacts.

### Train the Attack-Type Classifier

After the binary model and its required preprocessing artifacts have been generated:

```bash
python -m src.train_attack_type
```

The attack-type model is trained separately using the original attack labels.

### Run Prediction from the Command Line

```bash
python -m src.predict path\to\new_traffic.csv
```

Replace the example path with the location of your input CSV file.

## Results and Visualizations

Generated artifacts are organized under `results/`.

### Figures

- `class_distribution.png` — distribution of binary classes.
- `correlation_heatmap.png` — correlations among selected numeric features.
- `feature_importance.png` — feature importance for binary classification.
- `confusion_matrix.png` — binary classification results.
- `attack_type_confusion_matrix.png` — attack-type classification results.
- `attack_type_class_distribution.png` — distribution of attack categories.

### Metrics and Reports

The `results/metrics/` directory contains evaluation metrics, classification reports, training summaries, and feature-importance records.

The `results/selected_features/` directory contains feature-selection outputs and related analysis artifacts.

## Model Artifacts

The repository includes two trained models:

- `models/random_forest_cicids2017.joblib` — binary traffic classification.
- `models/random_forest_attack_type.joblib` — attack-type classification.

The models were trained on the project's selected CICIDS2017 features. Input compatibility and the expected preprocessing pipeline must be maintained when using the saved artifacts.

## Evaluation Metrics

The project evaluates classification performance using:

- **Accuracy:** Overall proportion of correctly classified records.
- **Precision:** Proportion of predicted positive cases that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Breakdown of correct and incorrect predictions by class.

For binary evaluation, the positive class is `MALICIOUS`. Attack-type evaluation also reports weighted metrics across the supported attack categories.

## Limitations

- CICIDS2017 is a benchmark dataset and does not directly represent all characteristics of modern 5G traffic.
- Results depend on the dataset, selected features, and training configuration.
- High benchmark scores do not guarantee equivalent performance in real-world deployments.
- The classifier recognizes patterns represented in its training data and may not correctly identify previously unseen attack patterns.
- SHAP explanations describe model behavior and should not be treated as definitive security evidence.
- This project is an academic prototype and should not be considered a replacement for a production-grade intrusion detection system.

## Future Enhancements

- Evaluate the system using additional 5G, IoT, and modern intrusion detection datasets.
- Perform controlled hyperparameter tuning and robustness testing.
- Assess generalization to unseen attack patterns.
- Optimize inference time and resource usage for deployment.
- Improve security monitoring and integration with network management systems.

## Conclusion

This project demonstrates a Random Forest-based approach to network traffic classification using CICIDS2017. The two-stage architecture combines binary malicious traffic detection with attack-type identification, while SHAP-based explainability and a Streamlit interface make predictions easier to inspect.

The implementation provides a foundation for further research into machine learning-based intrusion detection and its application to 5G security.



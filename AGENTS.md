# Project Instructions

- Use the official CIC/UNB CICIDS2017 dataset source where possible.
- Do not fabricate CICIDS2017 data, metrics, feature importances, or confusion matrices.
- The only classifier in this project is `sklearn.ensemble.RandomForestClassifier`.
- The project is binary classification only: `BENIGN` becomes `NORMAL`; every other label becomes `MALICIOUS`.
- Any fitted preprocessing must be learned from training data only and reused for test/new traffic.

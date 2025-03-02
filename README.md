# Binary Classification with XGBoost

## Overview
This repository demonstrates binary classification using the XGBoost algorithm on a diabetes dataset. The goal is to predict whether a patient has diabetes (`Outcome = 1`) or not (`Outcome = 0`).

## Dataset
- **Source**: [Kaggle - Diabetes Dataset](https://www.kaggle.com/datasets/asinow/diabetes-dataset?resource=download)
- **Features**: Age, BMI, Glucose, Blood Pressure, LDL, HDL, etc.
- **Target**: `Outcome` (0 = No Diabetes, 1 = Diabetes)

## Data Preprocessing
- Removed negative values from `LDL` and `HDL`.
- One-hot encoded `DietType`.
- Train-test split (80/20) with stratification.

## Handling Imbalanced Data
- Used `scale_pos_weight` to adjust for class imbalance.
- Applied SMOTE to create a balanced dataset for comparison.

## Model Training
- **Algorithm**: XGBoost
- **Evaluation Metric**: Log loss
- **Hyperparameters**:
  - `max_depth=6`
  - `learning_rate=0.1`
  - `scale_pos_weight` applied for imbalance
- **Training Time**: ~0.12 seconds

## Model Performance
- **Imbalanced Dataset Accuracy**: `99.95%`
- **Balanced Dataset (SMOTE) Accuracy**: `99.36%`
- **Confusion Matrix**: Visualized with Seaborn.

## Dependencies
```bash
pip install numpy pandas xgboost scikit-learn seaborn matplotlib imbalanced-learn
```

## Acknowledgments
- Dataset by Kaggle user `asinow`.
- XGBoost used for efficient classification.


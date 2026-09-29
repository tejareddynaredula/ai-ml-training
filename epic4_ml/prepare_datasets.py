from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer

# Folder where the datasets will be saved
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Regression dataset: Diabetes
diabetes = load_diabetes(as_frame=True)
diabetes_df = diabetes.data.copy()
diabetes_df["target"] = diabetes.target
diabetes_path = DATA_DIR / "diabetes_regression.csv"
diabetes_df.to_csv(diabetes_path, index=False)

# 2. Classification dataset: Breast Cancer
breast_cancer = load_breast_cancer(as_frame=True)
cancer_df = breast_cancer.data.copy()
cancer_df["target"] = breast_cancer.target
cancer_path = DATA_DIR / "breast_cancer_classification.csv"
cancer_df.to_csv(cancer_path, index=False)

print("Datasets saved successfully!")
print(f"Regression dataset: {diabetes_path}")
print(f"Rows: {len(diabetes_df)}, Columns: {len(diabetes_df.columns)}")
print(f"\nClassification dataset: {cancer_path}")
print(f"Rows: {len(cancer_df)}, Columns: {len(cancer_df.columns)}")

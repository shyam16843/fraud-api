"""
train_model.py
Run this script once to train the XGBoost model and save it.
Place your creditcard.csv in the same directory before running.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

print("=" * 50)
print("Fraud Detection Model Training")
print("=" * 50)

# Load dataset
CSV_PATH = "creditcard.csv"
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# SMOTEENN resampling
print("\nApplying SMOTEENN resampling...")
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
print(f"After resampling: {X_resampled.shape[0]} samples")

# Train XGBoost
print("\nTraining XGBoost classifier...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_resampled, y_resampled)

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precision (Fraud): {precision_score(y_test, y_pred):.4f}")

# Save model
os.makedirs("model", exist_ok=True)
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to model/fraud_model.pkl")
print("✅ You can now run the API with: docker-compose up")

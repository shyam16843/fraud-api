"""
train_model.py - Final version (no scaling, XGBoost handles it natively)
Run this to retrain and save the model.
Place creditcard.csv in the same directory before running.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("=" * 50)
print("Fraud Detection Model Training v3")
print("=" * 50)

CSV_PATH = "creditcard.csv"
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

# No scaling — XGBoost handles raw features natively
X = df.drop("Class", axis=1)
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# SMOTE oversampling
print("\nApplying SMOTE resampling...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_resampled.shape[0]} samples")
print(f"Fraud in training: {y_resampled.sum()} cases")

# Train XGBoost
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    use_label_encoder=False,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_resampled, y_resampled,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Evaluate
print("\n--- Evaluation ---")
y_prob = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

for threshold in [0.3, 0.4, 0.5]:
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\nThreshold {threshold}:")
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate", "Fraud"]))

# Save model only — no scaler needed
os.makedirs("model", exist_ok=True)
with open("model/fraud_model.bin", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved to model/fraud_model.bin")
print("\nSample fraud probabilities from test set:")
fraud_idx = y_test[y_test == 1].index[:5]
for idx in fraud_idx:
    row_pos = X_test.index.get_loc(idx)
    prob = y_prob[row_pos]
    print(f"  Known fraud → probability: {prob:.4f}")

"""
train_model.py - Improved version
Run this to retrain and save a better model.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("=" * 50)
print("Fraud Detection Model Training v2")
print("=" * 50)

CSV_PATH = "creditcard.csv"
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

# Scale Amount and Time
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop("Class", axis=1)
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# SMOTE oversampling (more reliable than SMOTEENN)
print("\nApplying SMOTE resampling...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_resampled.shape[0]} samples")
print(f"Fraud in training: {y_resampled.sum()} cases")

# Calculate scale_pos_weight for XGBoost
neg = (y_resampled == 0).sum()
pos = (y_resampled == 1).sum()
scale = neg / pos
print(f"\nscale_pos_weight: {scale:.2f}")

# Train XGBoost with better params
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,
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

# Evaluate at different thresholds
print("\n--- Evaluation ---")
y_prob = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

for threshold in [0.3, 0.4, 0.5]:
    y_pred = (y_prob >= threshold).astype(int)
    p = precision_score(y_test, y_pred, zero_division=0)
    print(f"\nThreshold {threshold}:")
    print(classification_report(y_test, y_pred,
          target_names=["Legitimate", "Fraud"]))

# Save model + scaler
os.makedirs("model", exist_ok=True)
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved to model/fraud_model.pkl")
print("✅ Scaler saved to model/scaler.pkl")
print("\nSample fraud probabilities from test set:")
fraud_idx = y_test[y_test == 1].index[:5]
for idx in fraud_idx:
    row_pos = X_test.index.get_loc(idx)
    prob = y_prob[row_pos]
    print(f"  Known fraud → probability: {prob:.4f}")

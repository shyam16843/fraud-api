import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("Training model inside Docker...")

df = pd.read_csv("sample_fraud.csv")
print(f"Dataset: {df.shape}, Fraud: {df['Class'].sum()}")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_res, y_res)

os.makedirs("model", exist_ok=True)
with open("model/fraud_model.dat", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved inside Docker")
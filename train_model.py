"""
train_model.py
--------------
Downloads the PIMA Indians Diabetes dataset, preprocesses it,
trains Logistic Regression (primary) + Random Forest (optional),
evaluates both models, and saves the best one as model.pkl.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("📥  Loading PIMA Indians Diabetes dataset …")

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

local_csv = os.path.join(os.path.dirname(__file__), "diabetes.csv")
url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

if os.path.exists(local_csv):
    # Load from local file (already has header row)
    df = pd.read_csv(local_csv)
    print(f"   Loaded from local file: {local_csv}")
else:
    # Fallback: download from internet (no header in that file)
    print("   Downloading from internet …")
    df = pd.read_csv(url, header=None, names=columns)
    print("   Download complete.")
print(f"   Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n🔧  Preprocessing …")

# Columns where 0 is physiologically impossible → replace with column median
zero_not_valid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_not_valid:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)
    print(f"   Replaced 0s in '{col}' with median {median_val:.2f}")

# Features / target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train / test split  (80 / 20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train size : {X_train.shape[0]} samples")
print(f"   Test  size : {X_test.shape[0]} samples")

# Feature scaling (critical for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 3. TRAIN LOGISTIC REGRESSION  (primary model)
# ─────────────────────────────────────────────
print("\n🤖  Training Logistic Regression …")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict(X_test_scaled)
lr_acc   = accuracy_score(y_test, lr_preds)
print(f"   Accuracy : {lr_acc*100:.2f}%")
print("\n   Classification Report:\n", classification_report(y_test, lr_preds))

# ─────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST  (optional / comparison)
# ─────────────────────────────────────────────
print("\n🌲  Training Random Forest …")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)          # RF doesn't need scaled data

rf_preds = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)
print(f"   Accuracy : {rf_acc*100:.2f}%")
print("\n   Classification Report:\n", classification_report(y_test, rf_preds))

# ─────────────────────────────────────────────
# 5. SAVE MODEL + SCALER
# ─────────────────────────────────────────────
# We save a dict with both models + scaler so Flask can use either
bundle = {
    "scaler"       : scaler,
    "lr_model"     : lr_model,
    "rf_model"     : rf_model,
    "feature_names": list(X.columns),
    "lr_accuracy"  : round(lr_acc * 100, 2),
    "rf_accuracy"  : round(rf_acc * 100, 2),
}

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"\n✅  Model bundle saved → {model_path}")
print("    Keys:", list(bundle.keys()))
print("\nDone! You can now start the Flask app with:  python app.py")

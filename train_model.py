# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------------
# 1. GENERATE SYNTHETIC DATASET
# -------------------------------

np.random.seed(42)
n = 4000

data = pd.DataFrame({
    "tenure_months": np.random.randint(1, 60, n),
    "monthly_recharge": np.random.randint(5, 100, n),
    "monthly_transactions": np.random.randint(1, 200, n),
    "previous_loans": np.random.randint(0, 5, n),
    "loan_amount": np.random.randint(10, 500, n),
})

# Create realistic default behavior
data["loan_default"] = (
    (data["loan_amount"] > data["monthly_recharge"] * 3) |
    (data["previous_loans"] >= 3)
).astype(int)

print("Dataset created successfully")

# -------------------------------
# 2. SPLIT FEATURES & TARGET
# -------------------------------

X = data.drop("loan_default", axis=1)
y = data["loan_default"]

# -------------------------------
# 3. TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. HANDLE IMBALANCED DATA
# -------------------------------

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -------------------------------
# 5. TRAIN XGBOOST MODEL
# -------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -------------------------------
# 6. EVALUATION
# -------------------------------

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# -------------------------------
# 7. SAVE MODEL
# -------------------------------

joblib.dump(model, "credit_model.pkl")

print("\nModel trained and saved as credit_model.pkl")
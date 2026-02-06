# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 2. LOAD DATA
# Path given by system
data_path = "E:\TASK 3\Task 3 and 4_Loan_Data.csv"
df = pd.read_csv(data_path)

print("First 5 rows:")
print(df.head())

# 3. BASIC DATA INSPECTION

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isna().sum())


# 4. DEFINE FEATURES & TARGET
# Assume target column is called 'default'
# (change if your file uses a different name)

TARGET = "default"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 6. FEATURE SCALING

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. TRAIN DECISION TREE

dt_model = DecisionTreeClassifier(
    max_depth=4,          # prevent overfitting
    min_samples_leaf=25,
    random_state=42
)

dt_model.fit(X_train_scaled, y_train)

# Visualize the tree

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,15))
plot_tree(dt_model, feature_names=X.columns, class_names=["No Default", "Default"], filled=True)
plt.show()


# 8. MODEL EVALUATION

y_pred = dt_model.predict(X_test_scaled)
y_prob = dt_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("AUC Score:", roc_auc_score(y_test, y_prob))

# 9. EXPECTED LOSS FUNCTION

RECOVERY_RATE = 0.10  # 10%

def expected_loss(loan_features: dict, loan_amount: float):
    """
    loan_features: dictionary of borrower features
    loan_amount: exposure at default (EAD)

    returns: (PD, Expected Loss)
    """

    # Convert to dataframe
    input_df = pd.DataFrame([loan_features])

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Predict probability of default
    pd_default = dt_model.predict_proba(input_scaled)[0][1]

    # Expected Loss formula
    el = pd_default * (1 - RECOVERY_RATE) * loan_amount

    return pd_default, el

# 10. EXAMPLE USAGE
# IMPORTANT: Sample borrower must contain ALL features from training data in the correct order
sample_borrower = {
    "customer_id": 123456,
    "credit_lines_outstanding": 2,
    "loan_amt_outstanding": 5000,
    "total_debt_outstanding": 15000,
    "income": 50000,
    "years_employed": 8,
    "fico_score": 650
}

loan_amount = 10000

pd_value, el_value = expected_loss(sample_borrower, loan_amount)

print("\n--- Sample Prediction ---")
print("Probability of Default:", round(pd_value, 4))
print("Expected Loss (€):", round(el_value, 2))

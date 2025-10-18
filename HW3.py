import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 0: Load and inspect the new dataset
df = pd.read_csv("Loan Eligibility Prediction.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print("\nData types:\n", df.dtypes)
print("\nFirst few rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Convert target to numeric (Y=1, N=0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# correlation
corr = df.corr(numeric_only=True)["Loan_Status"].sort_values(ascending=False)
print(corr)

# Step 1 Prep
# Drop ID (not useful for prediction)
df = df.drop(columns=["Customer_ID"])


# Encode categorical columns (turn text into numbers)
df = pd.get_dummies(df, columns=["Gender", "Married", "Education",
                                 "Self_Employed", "Property_Area"],
                    drop_first=True)

# Separate features (X) and target (y)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Split into train (70%), dev (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# Standardize numeric columns (makes model training more stable)
scaler = StandardScaler()
num_cols = ["Applicant_Income", "Coapplicant_Income",
            "Loan_Amount", "Loan_Amount_Term", "Credit_History"]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_dev[num_cols] = scaler.transform(X_dev[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("Train:", X_train.shape, "Dev:", X_dev.shape, "Test:", X_test.shape)

# Step 1.1
print("\nStep 1\n")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Train a Logistic Regression model with default settings
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate on the development set
y_pred_dev = model.predict(X_dev)
acc = accuracy_score(y_dev, y_pred_dev)
f1 = f1_score(y_dev, y_pred_dev, average="weighted")

print("Accuracy (dev):", round(acc, 4))
print("Weighted F1 (dev):", round(f1, 4))
 
# Step 1.2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Train a Support Vector Machine (default settings)
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate on the development set
y_pred_dev_svm = svm_model.predict(X_dev)
acc_svm = accuracy_score(y_dev, y_pred_dev_svm)
f1_svm = f1_score(y_dev, y_pred_dev_svm, average="weighted")

print("SVM Accuracy (dev):", round(acc_svm, 4))
print("SVM Weighted F1 (dev):", round(f1_svm, 4))

# Step 2.1
print("\nStep2\n")
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

# Define parameter grid (small, simple)
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1],
    "kernel": ["rbf"]
}

# Grid search with 5-fold cross-validation on the training data
grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring="f1_weighted")
grid.fit(X_train, y_train)

# Best model
best_svm = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate on development set
y_pred_dev_best = best_svm.predict(X_dev)
acc_best = accuracy_score(y_dev, y_pred_dev_best)
f1_best = f1_score(y_dev, y_pred_dev_best, average="weighted")

print("Tuned SVM Accuracy (dev):", round(acc_best, 4))
print("Tuned SVM Weighted F1 (dev):", round(f1_best, 4))

# Step 2.2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# Small parameter grid for regularization strength (C)
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs"]
}

# Grid search with 5-fold cross-validation
grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                    param_grid, cv=5, scoring="f1_weighted")
grid.fit(X_train, y_train)

# Best model
best_lr = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Evaluate on development set
y_pred_dev_best_lr = best_lr.predict(X_dev)
acc_best_lr = accuracy_score(y_dev, y_pred_dev_best_lr)
f1_best_lr = f1_score(y_dev, y_pred_dev_best_lr, average="weighted")

print("Tuned Logistic Regression Accuracy (dev):", round(acc_best_lr, 4))
print("Tuned Logistic Regression Weighted F1 (dev):", round(f1_best_lr, 4))


# Step 3 
print("\nStep 3\n")
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # Convert to pure numeric NumPy arrays
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            # Euclidean distance to all training samples
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Get indices of k nearest neighbors
            k_idx = np.argsort(distances)[:self.k]
            # Majority vote
            k_labels = self.y_train[k_idx]
            most_common = Counter(k_labels).most_common(1)[0][0]
            preds.append(most_common)
        return np.array(preds)

# ---- Tune k on the development set ----
best_k, best_acc, best_f1 = 0, 0, 0
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    y_pred_dev = knn.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred_dev)
    f1 = f1_score(y_dev, y_pred_dev, average="weighted")
    print(f"k={k} | Accuracy={acc:.4f} | F1={f1:.4f}")
    if f1 > best_f1:
        best_k, best_acc, best_f1 = k, acc, f1

print(f"\nBest k={best_k} | Accuracy={best_acc:.4f} | Weighted F1={best_f1:.4f}")

# Step 4
print("\nStep 4\n")
from sklearn.dummy import DummyClassifier
# --- 1. Most frequent strategy ---
dummy_mf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_mf.fit(X_train, y_train)
y_pred_mf = dummy_mf.predict(X_dev)
acc_mf = accuracy_score(y_dev, y_pred_mf)
f1_mf = f1_score(y_dev, y_pred_mf, average="weighted")

# --- 2. Stratified strategy ---
dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
dummy_strat.fit(X_train, y_train)
y_pred_strat = dummy_strat.predict(X_dev)
acc_strat = accuracy_score(y_dev, y_pred_strat)
f1_strat = f1_score(y_dev, y_pred_strat, average="weighted")

print("Most Frequent:\n       Accuracy:", round(acc_mf, 4), "| F1:", round(f1_mf, 4))
print("Stratified:\n       Accuracy:", round(acc_strat, 4), "| F1:", round(f1_strat, 4))

# Step 5
print("\nStep 5\n")

# Evaluate tuned SVM on the test set
y_pred_test_svm = best_svm.predict(X_test)
acc_svm_test = accuracy_score(y_test, y_pred_test_svm)
f1_svm_test = f1_score(y_test, y_pred_test_svm, average="weighted")

# Evaluate best KNN (k=5) on the test set 
best_knn = KNN(k=5)
best_knn.fit(X_train, y_train)
y_pred_test_knn = best_knn.predict(X_test)
acc_knn_test = accuracy_score(y_test, y_pred_test_knn)
f1_knn_test = f1_score(y_test, y_pred_test_knn, average="weighted")

print("Tuned SVM → Accuracy:", round(acc_svm_test, 4), "| F1:", round(f1_svm_test, 4))
print("Best KNN  → Accuracy:", round(acc_knn_test, 4), "| F1:", round(f1_knn_test, 4))

# Baseline results from Step 4 
print("\nBaselines:")
print("Most Frequent → Accuracy: 0.6848 | F1: 0.5567")
print("Stratified    → Accuracy: 0.5761 | F1: 0.574")
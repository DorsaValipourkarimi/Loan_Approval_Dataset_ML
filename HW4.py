import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Step 0: Load and inspect the new dataset
print("\nDataset:\n")

df = pd.read_csv("Loan Eligibility Prediction.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
#print("\nData types:\n", df.dtypes)
#print("\nFirst few rows:\n", df.head())
#print("\nMissing values:\n", df.isnull().sum())

# Convert target to numeric (Y=1, N=0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Step 1 Prep
print("\nStep 1\n")

# Drop ID (not useful for prediction)
df = df.drop(columns=["Customer_ID"])


# Encode categorical columns (turn text into numbers)
df = pd.get_dummies(df, columns=["Gender", "Married", "Education",
                                 "Self_Employed", "Property_Area"],
                    drop_first=True)

# Separate features (X) and target (y)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

from sklearn.model_selection import train_test_split

# Step 1: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# Step 2
print("\nStep 2\n")

# helper function: make simple v-folds manually
# Split y into n folds (stratified) for manual cross-validation.
def v_fold_split(y, n_folds=5, random_state=42):
    np.random.seed(random_state)
    y = np.array(y)
    folds = []

    # Get indices for each class (0s and 1s)
    class0 = np.where(y == 0)[0]
    class1 = np.where(y == 1)[0]

    # Shuffle each class
    np.random.shuffle(class0)
    np.random.shuffle(class1)

    # Split each class into n parts
    chunks0 = np.array_split(class0, n_folds)
    chunks1 = np.array_split(class1, n_folds)

    # Combine class chunks to make stratified folds
    for i in range(n_folds):
        val_idx = np.concatenate((chunks0[i], chunks1[i]))
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        folds.append((train_idx, val_idx))
    return folds

# Perform v-fold cross validation on training data using SVM
def v_fold_svm(X_train, y_train, n_folds=5, C=1.0, kernel='rbf', gamma='scale'):
    folds = v_fold_split(y_train, n_folds=n_folds)
    scores = []

    for i, (train_idx, val_idx) in enumerate(folds, 1):
        # Split data into training and validation sets
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train SVM
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_tr, y_tr)

        # Predict and score
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        scores.append(f1)
        print(f"Fold {i}: F1 = {f1:.4f}")

    print(f"\nAverage F1 across {n_folds} folds: {np.mean(scores):.4f}")
    return scores, np.mean(scores)

# Run
fold_scores, avg_f1 = v_fold_svm(X_train, y_train, n_folds=5)
std_f1 = float(np.std(fold_scores, ddof=1))
print(f"Std of F1 across folds: {std_f1:.4f}")


# Step 3
print("\nStep 3\n")

# Manual grid search over two SVM hyperparameters using our v-fold CV
# Grid search: loops over C and gamma values for RBF SVM.
# Uses our v-fold CV (v_fold_svm) to compute mean F1 for each combo.
# Returns: all results and the best setting.

def grid_search(X_train, y_train, C_list, gamma_list, n_folds=5):
   
    results = []
    best = {"C": None, "gamma": None, "mean_f1": -np.inf, "std_f1": None, "scores": None}

    for C in C_list:
        for gamma in gamma_list:
            scores, mean_f1 = v_fold_svm(
                X_train, y_train, n_folds=n_folds, C=C, kernel='rbf', gamma=gamma
            )
            std_f1 = float(np.std(scores, ddof=1))
            results.append({
                "C": C,
                "gamma": gamma,
                "scores": scores,
                "mean_f1": float(mean_f1),
                "std_f1": std_f1
            })
            print(f"[Grid] C={C}, gamma={gamma} → mean F1={mean_f1:.4f} (±{std_f1:.4f})")

            # Update best model
            if (mean_f1 > best["mean_f1"]) or (
                np.isclose(mean_f1, best["mean_f1"]) and std_f1 < (best["std_f1"] or np.inf)
            ):
                best = {"C": C, "gamma": gamma, "mean_f1": float(mean_f1),
                        "std_f1": std_f1, "scores": scores}

    print("\nBest combination:")
    print(f"  C = {best['C']}, gamma = {best['gamma']} → mean F1 = {best['mean_f1']:.4f} (±{best['std_f1']:.4f})")
    return results, best


# --- Define a small grid ---
C_list     = [0.01, 0.1, 1, 10, 100]
gamma_list = ["scale", 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


# Run Step 3
grid_results, best_model = grid_search(X_train, y_train, C_list, gamma_list, n_folds=5)


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Step 4
print("\nStep 4\n")

# Scale features (fit on training, apply to both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train best model on full training data
best_C = 10
best_gamma = 0.003

final_model = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
final_model.fit(X_train_scaled, y_train)

# Evaluate on training data
train_preds = final_model.predict(X_train_scaled)
train_f1 = f1_score(y_train, train_preds)

# Evaluate on test data (held-out set)
test_preds = final_model.predict(X_test_scaled)
test_f1 = f1_score(y_test, test_preds)

print(f"Training F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Difference: {train_f1 - test_f1:.4f}")

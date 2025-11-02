import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Step 0: Load and inspect
print("\nDataset:\n")

df = pd.read_csv("Loan Eligibility Prediction.csv")
print("Shape:", df.shape)
print("Columns:", df.columns)

# Convert target to numeric (Y=1, N=0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})


# Step 1: Prep + split
print("\nStep 1\n")

# Drop ID (not useful for prediction)
df = df.drop(columns=["Customer_ID"])

# Encoding categorical columns
df = pd.get_dummies(
    df,
    columns=["Gender", "Married", "Education", "Self_Employed", "Property_Area"],
    drop_first=True
)

# Separate features and target
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Step 2: Manual v-fold CV (with per-fold scaling)
print("\nStep 2\n")

#Make simple stratified folds for manual CV.
def v_fold_split(y, n_folds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    y = np.array(y)
    folds = []

    class0 = np.where(y == 0)[0]
    class1 = np.where(y == 1)[0]
    rng.shuffle(class0)
    rng.shuffle(class1)

    chunks0 = np.array_split(class0, n_folds)
    chunks1 = np.array_split(class1, n_folds)

    for i in range(n_folds):
        val_idx = np.concatenate((chunks0[i], chunks1[i]))
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        folds.append((train_idx, val_idx))
    return folds

# Manual v-fold CV using SVM.
# Per-fold StandardScaler (fit on train-fold only; transform train/val).
# Reports per-fold F1 and mean F1.
def v_fold_svm(X_train, y_train, n_folds=5, C=1.0, kernel='rbf', gamma='scale', random_state=42):
    folds = v_fold_split(y_train, n_folds=n_folds, random_state=random_state)
    scores = []

    for i, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_vals = scaler.transform(X_val)

        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_trs, y_tr)
        preds = model.predict(X_vals)

        f1 = f1_score(y_val, preds)
        scores.append(f1)
        print(f"Fold {i}: F1 = {f1:.4f}")

    mean_f1 = float(np.mean(scores))
    print(f"\nAverage F1 across {n_folds} folds: {mean_f1:.4f}")
    return scores, mean_f1

# Run Step 2 baseline CV with default SVM 
fold_scores, avg_f1 = v_fold_svm(X_train, y_train, n_folds=5, C=1.0, kernel='rbf', gamma='scale', random_state=42)
std_f1 = float(np.std(fold_scores, ddof=1))
print(f"Std of F1 across folds: {std_f1:.4f}")

# Step 3: From-scratch grid search (uses scaled CV above)
print("\nStep 3\n")

# Manual grid search over C and gamma for RBF SVM using our v-fold CV with per-fold scaling.
# Returns (results, best_dict).
def grid_search(X_train, y_train, C_list, gamma_list, n_folds=5, random_state=42):    
    results = []
    best = {"C": None, "gamma": None, "mean_f1": -np.inf, "std_f1": None, "scores": None}

    for C in C_list:
        for gamma in gamma_list:
            scores, mean_f1 = v_fold_svm(
                X_train, y_train, n_folds=n_folds, C=C, kernel='rbf', gamma=gamma, random_state=random_state
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

            if (mean_f1 > best["mean_f1"]) or (
                np.isclose(mean_f1, best["mean_f1"]) and std_f1 < (best["std_f1"] or np.inf)
            ):
                best = {"C": C, "gamma": gamma, "mean_f1": float(mean_f1),
                        "std_f1": std_f1, "scores": scores}

    print("\nBest combination:")
    print(f"  C = {best['C']}, gamma = {best['gamma']} → mean F1 = {best['mean_f1']:.4f} (±{best['std_f1']:.4f})")
    return results, best

# Define a grid 
C_list     = [0.01, 0.1, 1, 10, 100]
gamma_list = ["scale", 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

# Run Step 3
grid_results, best_model = grid_search(X_train, y_train, C_list, gamma_list, n_folds=5, random_state=42)


# Step 4: Final train on full training set + test evaluation
print("\nStep 4\n")

# Scale with a single scaler fit on the full training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Use the best hyperparameters found by the scaled CV grid search
best_C = best_model["C"]
best_gamma = best_model["gamma"]

final_model = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
final_model.fit(X_train_scaled, y_train)

# Evaluate on training data
train_preds = final_model.predict(X_train_scaled)
train_f1 = f1_score(y_train, train_preds)

# Evaluate on held-out test data
test_preds = final_model.predict(X_test_scaled)
test_f1 = f1_score(y_test, test_preds)

print(f"Best params → C={best_C}, gamma={best_gamma}")
print(f"Training F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Difference: {train_f1 - test_f1:.4f}")

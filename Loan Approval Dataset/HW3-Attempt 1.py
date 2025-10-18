import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Step 0
# Load dataset
df = pd.read_csv("loan_approval.csv.xls")
'''
print(df.shape)
print(df.columns)
print(df.head())
print(df["loan_approved"].value_counts())
'''
# Drop 'name' (not useful) and convert target to numeric
df = df.drop(columns=["name"])
df["loan_approved"] = df["loan_approved"].astype(int)

# One-hot encode city (turn text into numbers)
df = pd.get_dummies(df, columns=["city"], drop_first=True)

# Split data (70% train, 15% dev, 15% test)
X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]
# This separates 70% for training and 30% for temporary data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
# This split that 600-sample “temporary” set in half
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Scale numeric features for better model performance
scaler = StandardScaler()
num_cols = ["income", "credit_score", "loan_amount", "years_employed", "points"]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_dev[num_cols] = scaler.transform(X_dev[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train Logistic Regression with default settings
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

'''# Evaluate on development set
y_pred_dev = model.predict(X_dev)
print("Accuracy (dev):", accuracy_score(y_dev, y_pred_dev))
print("F1 (weighted, dev):", f1_score(y_dev, y_pred_dev, average="weighted"))

y_pred_test = model.predict(X_test)
print("Accuracy (test):", accuracy_score(y_test, y_pred_test))
print("F1 (weighted, test):", f1_score(y_test, y_pred_test, average="weighted"))

df_corr = df.copy()
df_corr["loan_approved"] = df_corr["loan_approved"].astype(int)
print(df_corr.corr()["loan_approved"].sort_values(ascending=False))
'''

# Convert target to int for plotting
df["loan_approved"] = df["loan_approved"].astype(int)

# Create the scatter plot of credit_score vs points
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="credit_score",
    y="points",
    hue="loan_approved",
    palette={0: "red", 1: "green"},
    alpha=0.6
)
plt.title("Loan Approval: Points vs Credit Score")
plt.xlabel("Credit Score")
plt.ylabel("Points")
plt.legend(title="Approved", labels=["No", "Yes"])

# Save instead of showing
plt.savefig("loan_approval_plot.png", dpi=150)
print("✅ Plot saved as 'loan_approval_plot.png' — open it in your workspace to view.")

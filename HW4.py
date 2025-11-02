# Step 0: Load and inspect the new dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 0: Load and inspect the new dataset
df = pd.read_csv("Loan Eligibility Prediction.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print("\nData types:\n", df.dtypes)
print("\nFirst few rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Convert target to numeric (Y=1, N=0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

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
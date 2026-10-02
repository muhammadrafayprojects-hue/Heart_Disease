# Heart Disease Dataset Analysis
# This script loads the dataset and prints useful information for beginners.

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
# Make sure the file heart.csv is in the same folder as this script
# If not, update the path accordingly.
df = pd.read_csv("heart.csv")

# Print the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())
print()

# Print the shape of the dataset: (rows, columns)
print("Dataset shape:")
print(df.shape)
print()

# Print all column names
print("Column names:")
print(df.columns)
print()

# Check for missing values in each column
print("Missing values in each column:")
print(df.isnull().sum())
print()

# Print basic statistical summary of the dataset
print("Basic statistics:")
print(df.describe())
print()

# Print unique values for selected categorical columns
# These columns are important for understanding the dataset better.
columns_to_check = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "target",
]

for column in columns_to_check:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print()


# GRAPHS / DATA VISUALIZATION

# 1. Target Distribution
plt.figure(figsize=(6, 4))

df["target"].value_counts().sort_index().plot(kind="bar")

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.show()


# 2. Age Distribution
plt.figure(figsize=(7, 4))

plt.hist(df["age"], bins=10)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# 3. Chest Pain Type vs Target
plt.figure(figsize=(7, 4))

pd.crosstab(df["cp"], df["target"]).plot(kind="bar")

plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.legend(["No Disease", "Disease"])
plt.show()


# 4. Cholesterol Distribution
plt.figure(figsize=(7, 4))

plt.hist(df["chol"], bins=10)

plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Number of Patients")
plt.show()


# 5. Resting Blood Pressure Distribution
plt.figure(figsize=(7, 4))

plt.hist(df["trestbps"], bins=10)

plt.title("Resting Blood Pressure Distribution")
plt.xlabel("Blood Pressure")
plt.ylabel("Number of Patients")
plt.show()


# 6. Maximum Heart Rate Distribution
plt.figure(figsize=(7, 4))

plt.hist(df["thalach"], bins=10)

plt.title("Maximum Heart Rate Distribution")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Number of Patients")
plt.show()
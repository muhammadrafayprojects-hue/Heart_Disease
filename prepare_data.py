# Heart Disease Dataset Preparation
# This script loads the dataset, separates features and target,
# and splits the data into training and testing sets.

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
# Make sure heart.csv is in the same folder as this script
# If not, update the path accordingly.
df = pd.read_csv("heart.csv")

# Separate features (X) and target (y)
# The target column is the dependent variable we want to predict.
X = df.drop(columns=["target"])
y = df["target"]

# Split the data into training and testing sets
# 80% training, 20% testing
# random_state=42 makes the split reproducible
# stratify=y keeps the class distribution similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Print the shapes of the data
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print()

# Print all feature columns
print("Feature columns:")
print(X.columns.tolist())

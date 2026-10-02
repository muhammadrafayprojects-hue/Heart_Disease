# Heart Disease Prediction Model
# This script loads the dataset, trains a Random Forest classifier,
# evaluates its performance, and saves the trained model.

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# Make sure heart.csv is in the same folder as this script
# If not, update the path accordingly.
df = pd.read_csv("heart.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print()

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()

# Compute confusion matrix values manually for easier interpretation
cm = confusion_matrix(y_test, y_pred)
true_negative = cm[0][0]
true_positive = cm[1][1]
false_positive = cm[0][1]
false_negative = cm[1][0]

print("True Negative:", true_negative)
print("True Positive:", true_positive)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print()

# Explain target labels
print("Target values:")
print("0 = No Heart Disease")
print("1 = Heart Disease")
print()

# Save the trained model to a .pkl file
joblib.dump(model, "heart_disease_model.pkl")
print("Model successfully saved as heart_disease_model.pkl")

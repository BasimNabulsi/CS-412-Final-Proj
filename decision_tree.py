# Import necessary libraries
import ssl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# from ucimlrepo import fetch_ucirepo

# Load the dataset from a local CSV file
data = pd.read_csv("data.csv")

# Display first few rows of the data
print("First few rows of the dataset:")
print(data.head())

# Display descriptive statistics for the data
print("\nDescriptive statistics:")
print(data.describe())

# Separate features and target variable
X = data.drop('DEATH_EVENT', axis=1)  # Features
y = data['DEATH_EVENT']  # Target variable (binary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model's performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'\nDecision Tree Accuracy: {accuracy_dt:.2f}')

# Print classification report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Classification
# Make predictions on the entire dataset
y_pred_all = dt_model.predict(X)

# Evaluate the model's performance on the entire dataset
accuracy_all = accuracy_score(y, y_pred_all)
classification_report_all = classification_report(y, y_pred_all)

print(f'\nDecision Tree Accuracy (on entire dataset): {accuracy_all:.2f}')
print("\nDecision Tree Classification Report (on entire dataset):")
print(classification_report_all)

# Risk Stratification
# Predict probabilities of heart failure for each patient
y_probabilities = dt_model.predict_proba(X)[:, 1]

# Define thresholds for high-risk and low-risk groups (you can adjust these)
high_risk_threshold = 0.5
low_risk_threshold = 0.2

# Categorize patients into high-risk and low-risk groups
high_risk_patients = data[y_probabilities >= high_risk_threshold]
low_risk_patients = data[y_probabilities <= low_risk_threshold]

# Print the number of patients in each risk group
print("\nNumber of patients in each risk group:")
print(f"High-risk patients: {len(high_risk_patients)}")
print(f"Low-risk patients: {len(low_risk_patients)}")

# Feature Importance Analysis
# Get feature importances from the trained decision tree
feature_importance = dt_model.feature_importances_
feature_names = X.columns

# Create a DataFrame to store feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print("\nFeature Importance:")
print(importance_df)

# # Visualize the decision tree structure
# tree_rules = export_text(dt_model, feature_names=list(X.columns))
# print("\nDecision Tree Structure:")
# print(tree_rules)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['No Heart Failure', 'Heart Failure'])
plt.title('Decision Tree Visualization')
plt.show()

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
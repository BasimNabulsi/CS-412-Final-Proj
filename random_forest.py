# Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from a local CSV file
data = pd.read_csv("data.csv")

# Display first few rows of the data
print("First few rows of the dataset:")
print(data.head())

# Display descriptive statistics for the data
print("\nDescriptive statistics:")
print(data.describe())

# Plot pairplot to see relationships between features and target variable
sns.pairplot(data, hue='DEATH_EVENT')  # Change 'death_event' to 'DEATH_EVENT'
# Get the path to the Downloads folder on Mac
downloads_folder = os.path.expanduser("~/Downloads")

# Save the plot to the Downloads folder
plt.savefig(os.path.join(downloads_folder, 'pairplot.png'))

plt.show()

# Plot heatmap to see correlations between features
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Separate features and target variable
X = data.drop('DEATH_EVENT', axis=1)  # Features
y = data['DEATH_EVENT']  # Target variable (binary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.2f}')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Analyze feature importance
feature_importance = rf_model.feature_importances_
feature_names = X.columns

# Display feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)


print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


def get_user_input():
    user_data = {}

    user_data['age'] = float(input("Enter age (years): "))
    user_data['anaemia'] = int(input("Do you have anemia? (1 for yes, 0 for no): "))
    user_data['creatinine_phosphokinase'] = float(input("Enter creatinine phosphokinase level (mcg/L): "))
    user_data['diabetes'] = int(input("Do you have diabetes? (1 for yes, 0 for no): "))
    user_data['ejection_fraction'] = float(input("Enter ejection fraction (%): "))
    user_data['high_blood_pressure'] = int(input("Do you have high blood pressure? (1 for yes, 0 for no): "))
    user_data['platelets'] = float(input("Enter platelets level (kiloplatelets/mL): "))
    user_data['serum_creatinine'] = float(input("Enter serum creatinine level (mg/dL): "))
    user_data['serum_sodium'] = float(input("Enter serum sodium level (mEq/L): "))
    user_data['sex'] = int(input("Enter sex (1 for male, 0 for female): "))
    user_data['smoking'] = int(input("Do you smoke? (1 for yes, 0 for no): "))
    user_data['time'] = int(input("Enter follow-up period (days): "))
    
    user_df = pd.DataFrame(user_data, index=[0])
    
    return user_df


user_input = get_user_input()

user_input_df = pd.DataFrame(user_input, index=[0])

prediction_proba = rf_model.predict_proba(user_input_df)
heart_failure_probability = prediction_proba[0][1]
heart_failure_probability_percent = heart_failure_probability * 100

print("Based on the provided information, there is a {:.2f}% probability of experiencing heart failure.".format(heart_failure_probability_percent))

# user_input = {
#     'age': 70,  # Age in years
#     'anaemia': 1,  # Presence of anaemia (1 for yes, 0 for no)
#     'creatinine_phosphokinase': 200,  # Level of creatinine phosphokinase in mcg/L
#     'diabetes': 0,  # Presence of diabetes (1 for yes, 0 for no)
#     'ejection_fraction': 30,  # Ejection fraction percentage
#     'high_blood_pressure': 1,  # Presence of high blood pressure (1 for yes, 0 for no)
#     'platelets': 300000,  # Platelet count in kiloplatelets/mL
#     'serum_creatinine': 1.2,  # Level of serum creatinine in mg/dL
#     'serum_sodium': 135,  # Level of serum sodium in mEq/L
#     'sex': 1,  # Gender (1 for male, 0 for female)
#     'smoking': 0,  # Smoking status (1 for yes, 0 for no)
#     'time': 150  # Follow-up period in days
# }

# user_input = {
#     'age': 75,  # Age in years
#     'anaemia': 1,  # Presence of anaemia (1 for yes, 0 for no)
#     'creatinine_phosphokinase': 300,  # Level of creatinine phosphokinase in mcg/L
#     'diabetes': 1,  # Presence of diabetes (1 for yes, 0 for no)
#     'ejection_fraction': 25,  # Ejection fraction percentage
#     'high_blood_pressure': 1,  # Presence of high blood pressure (1 for yes, 0 for no)
#     'platelets': 200000,  # Platelet count in kiloplatelets/mL
#     'serum_creatinine': 1.5,  # Level of serum creatinine in mg/dL
#     'serum_sodium': 135,  # Level of serum sodium in mEq/L
#     'sex': 1,  # Gender (1 for male, 0 for female)
#     'smoking': 0,  # Smoking status (1 for yes, 0 for no)
#     'time': 100  # Follow-up period in days
# }
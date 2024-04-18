# Import necessary libraries
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

# Display feature importance
print("\nFeature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Function to get user input for clinical data
def get_user_input():
    # Initialize a dictionary to hold the user's data
    user_data = {}
    
    # Prompt the user for input and convert to the appropriate data type
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
    
    # Convert the user data to a DataFrame
    user_df = pd.DataFrame(user_data, index=[0])
    
    return user_df

# Get user input
user_input = get_user_input()

# Make predictions using the trained Random Forest model
prediction = rf_model.predict(user_input)

# Provide risk rating based on the model's prediction
if prediction == 1:
    print("Based on the provided information, you may have a high risk of heart failure. Please consult a medical professional.")
else:
    print("Based on the provided information, you have a low risk of heart failure.")
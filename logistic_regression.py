from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_failure_clinical_records = fetch_ucirepo(id=519) 
  
# data (as pandas dataframes) 
X = heart_failure_clinical_records.data.features 
y = heart_failure_clinical_records.data.targets 
  
# metadata 
# print(heart_failure_clinical_records.metadata) 
  
# variable information 
# print(heart_failure_clinical_records.variables) 

#-----------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['death_event']), df['death_event'], test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the logistic regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)

# Evaluating the model
train_score = log_reg_model.score(X_train_scaled, y_train)
test_score = log_reg_model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_score}")
print(f"Testing accuracy: {test_score}")




import numpy as np

# Get feature names
feature_names = heart_failure_clinical_records.variables['name'][:-1]  # Exclude the target variable

# Get coefficients
coefficients = log_reg_model.coef_[0]

# Calculate absolute coefficients for visualization
abs_coefficients = np.abs(coefficients)

# Sort feature names and coefficients by absolute coefficient values
sorted_indices = np.argsort(abs_coefficients)[::-1]  # Reverse order to get highest coefficients first
sorted_feature_names = feature_names[sorted_indices]
sorted_coefficients = abs_coefficients[sorted_indices]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.plot(range(len(sorted_feature_names)), sorted_coefficients, marker='o', color='skyblue', linestyle='-')
plt.xticks(range(len(sorted_feature_names)), sorted_feature_names, rotation=45, ha='right')
plt.ylabel('Coefficient Magnitude')
plt.xlabel('Feature')
plt.title('Death Event Prediction Based on Each Category')
plt.grid(True)
plt.show()


feature_target_correlation = df.corr()['death_event'].drop('death_event')

# Plot bar plot of correlation coefficients
plt.figure(figsize=(10, 6))
feature_target_correlation.plot(kind='bar', color='skyblue')
plt.title('Correlation with Death Event')
plt.xlabel('Feature')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

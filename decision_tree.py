import ssl
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Set the default SSL context to not verify certificates
ssl._create_default_https_context = ssl._create_unverified_context

# Fetch the dataset
heart_failure_clinical_records = fetch_ucirepo(id=519)

# Separate features and target variable
X = heart_failure_clinical_records.data.features
y = heart_failure_clinical_records.data.targets

# Metadata
print(heart_failure_clinical_records.metadata)

# Variable information
print(heart_failure_clinical_records.variables)

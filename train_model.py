import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the training data
data_path = 'cleaned_plant_disease_dataset.csv'
data = pd.read_csv(data_path)

# Print the column names for verification
print(data.columns)

# Assuming 'treatment_suggestions' is the target column, update as needed
target_column = 'treatment_suggestions'

# Update the feature columns
feature_columns = ['disease', 'plants', 'region', 'severity']

# Ensure all feature columns are in the dataset
if not all(col in data.columns for col in feature_columns):
    raise KeyError("One or more feature columns are not in the dataset")

X = data[feature_columns]
y = data[target_column]

# Convert categorical variables to numeric
X = pd.get_dummies(X)

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
model_path = 'best_random_forest_model.pkl'
joblib.dump(model, model_path)

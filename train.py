import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("glof_dataset.csv")

# Drop rows with missing target values
df.dropna(subset=['Impact'], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Driver_lake', 'Driver_GLOF', 'Mechanism', 'Impact_type', 'Repeat', 'Region_RGI', 'Impact']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save encoders for later use

# Select features and target
X = df.drop(columns=['Impact'])
y = df['Impact']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train XGBoost Classifier
xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import pickle

# Save the trained model
with open("glof_xgb_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)

# Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

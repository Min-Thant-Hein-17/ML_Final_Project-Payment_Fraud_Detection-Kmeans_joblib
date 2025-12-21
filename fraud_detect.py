
# train_model.py
import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# --- 1) Load dataset ---
fd = pd.read_csv("dataset/luxury_cosmetics_fraud_analysis_2025.csv")

# --- 2) Feature engineering ---
# Time -> continuous hour (H + M/60)
time_objs = pd.to_datetime(fd['Transaction_Time'], format='%H:%M:%S')
fd['Time_Continuous'] = time_objs.dt.hour + time_objs.dt.minute / 60.0

# Date -> day-of-week (Mon=0..Sun=6)
fd['Day_of_Week'] = pd.to_datetime(fd['Transaction_Date']).dt.dayofweek

# --- 3) Drop identifiers / label columns (don’t use labels for clustering) ---
cols_to_drop = [
    'Transaction_ID','Customer_ID','Fraud_Flag','IP_Address',
    'Transaction_Date','Transaction_Time'
]
fd_cleaned = fd.drop(columns=[c for c in cols_to_drop if c in fd.columns])

# --- 4) Define the ONLY features we will train on (freeze schema) ---
EXPECTED_FEATURES = [
    'Purchase_Amount', 'Customer_Age', 'Footfall_Count',
    'Time_Continuous', 'Day_of_Week',
    'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
]

# Ensure all expected columns exist – if any are missing, create nulls (imputer will handle)
for col in EXPECTED_FEATURES:
    if col not in fd_cleaned.columns:
        fd_cleaned[col] = np.nan

# --- 5) Preprocessors ---
num_cols = ['Purchase_Amount','Customer_Age','Footfall_Count','Time_Continuous','Day_of_Week']
cat_cols = ['Customer_Loyalty_Tier','Payment_Method','Product_Category']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # IMPORTANT: ignore unseen categories at inference
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='drop')

# --- 6) Model pipeline ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('cluster_model', kmeans)
])

# Fit
model.fit(fd_cleaned[EXPECTED_FEATURES])

# --- 7) Save model together with the expected schema ---
artifact = {
    'model': model,
    'expected_features': EXPECTED_FEATURES
}
with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("✅ Trained and saved fraud_detection_model.pkl with expected schema.")

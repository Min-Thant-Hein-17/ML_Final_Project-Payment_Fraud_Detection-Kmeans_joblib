
# train_model.py
# Purpose: Train an unsupervised K-Means clustering pipeline for payment transactions,
#          perform feature engineering, auto-select K via silhouette, and save artifacts via joblib/pickle.

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------- Configuration -----------------
DATA_PATH = os.getenv("DATA_PATH", "dataset/luxury_cosmetics_fraud_analysis_2025.csv")
ARTIFACT_JOBLIB = os.getenv("ARTIFACT_JOBLIB", "fraud_detection_model.joblib")
ARTIFACT_PKL = os.getenv("ARTIFACT_PKL", "fraud_detection_model.pkl")

EXPECTED_FEATURES = [
    'Purchase_Amount', 'Customer_Age', 'Footfall_Count',
    'Time_Continuous', 'Day_of_Week',
    'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
]

# ----------------- Load and Feature Engineering -----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place a CSV there with the required columns.")

fd = pd.read_csv(DATA_PATH)

# Derive Time_Continuous from HH:MM:SS and Day_of_Week from date
if 'Transaction_Time' in fd.columns:
    time_objs = pd.to_datetime(fd['Transaction_Time'], format='%H:%M:%S', errors='coerce')
    fd['Time_Continuous'] = time_objs.dt.hour + (time_objs.dt.minute / 60.0)
elif 'Time_Continuous' not in fd.columns:
    # If absent, fill with median or zeros to avoid failure
    fd['Time_Continuous'] = 12.0

if 'Transaction_Date' in fd.columns:
    fd['Day_of_Week'] = pd.to_datetime(fd['Transaction_Date'], errors='coerce').dt.dayofweek
elif 'Day_of_Week' not in fd.columns:
    fd['Day_of_Week'] = 0

# Drop identifiers/labels (if present)
cols_to_drop = ['Transaction_ID','Customer_ID','Fraud_Flag','IP_Address','Transaction_Date','Transaction_Time']
fd_cleaned = fd.drop(columns=[c for c in cols_to_drop if c in fd.columns])

# Guarantee all expected columns exist
for col in EXPECTED_FEATURES:
    if col not in fd_cleaned.columns:
        fd_cleaned[col] = np.nan

# ----------------- Preprocessor -----------------
num_cols = ['Purchase_Amount','Customer_Age','Footfall_Count','Time_Continuous','Day_of_Week']
cat_cols = ['Customer_Loyalty_Tier','Payment_Method','Product_Category']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Using sparse_output=False (scikit-learn >=1.2) for dense arrays; pin version in requirements
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ],
    remainder='drop'
)

# ----------------- K auto-selection via silhouette -----------------
X = fd_cleaned[EXPECTED_FEATURES]
X_proc = preprocessor.fit_transform(X)

best_k = None
best_sil = -1.0
best_model = None
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X_proc)
    sil = silhouette_score(X_proc, labels)
    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_model = km

# Build final pipeline with preprocessor + best KMeans
model = Pipeline(steps=[('preprocess', preprocessor), ('cluster_model', best_model)])
model.fit(X)

# Persist artifacts via joblib and pickle
artifact = {
    'model': model,
    'expected_features': EXPECTED_FEATURES,
    'best_k': best_k,
}
joblib.dump(artifact, ARTIFACT_JOBLIB)
with open(ARTIFACT_PKL, 'wb') as f:
    pickle.dump(artifact, f)

# Optional: preview clustered data (first 200 rows)
preds = model.predict(X)
preview = fd_cleaned.copy()
preview['cluster'] = preds
preview.head(200).to_csv('clustered_preview.csv', index=False)

print(f"✅ Training complete. Best K = {best_k} (silhouette={best_sil:.3f}).")
print(f"💾 Saved: {ARTIFACT_JOBLIB} and {ARTIFACT_PKL}")

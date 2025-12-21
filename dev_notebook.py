
# dev_notebook.py
# A compact development script to explore the dataset, run preprocessing, and visualize elbow & silhouette.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = os.getenv("DATA_PATH", "dataset/luxury_cosmetics_fraud_analysis_2025.csv")

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# Feature engineering (same as training)
if 'Transaction_Time' in df.columns:
    time_objs = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S', errors='coerce')
    df['Time_Continuous'] = time_objs.dt.hour + (time_objs.dt.minute / 60.0)
else:
    df['Time_Continuous'] = 12.0

if 'Transaction_Date' in df.columns:
    df['Day_of_Week'] = pd.to_datetime(df['Transaction_Date'], errors='coerce').dt.dayofweek
else:
    df['Day_of_Week'] = 0

cols_to_drop = ['Transaction_ID','Customer_ID','Fraud_Flag','IP_Address','Transaction_Date','Transaction_Time']
df_cleaned = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

EXPECTED_FEATURES = [
    'Purchase_Amount', 'Customer_Age', 'Footfall_Count',
    'Time_Continuous', 'Day_of_Week',
    'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
]
for col in EXPECTED_FEATURES:
    if col not in df_cleaned.columns:
        df_cleaned[col] = np.nan

num_cols = ['Purchase_Amount','Customer_Age','Footfall_Count','Time_Continuous','Day_of_Week']
cat_cols = ['Customer_Loyalty_Tier','Payment_Method','Product_Category']

num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_cols),('cat', cat_transformer, cat_cols)])

X = df_cleaned[EXPECTED_FEATURES]
X_proc = preprocessor.fit_transform(X)

Ks = list(range(2, 9))
inertias, sils = [], []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X_proc)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_proc, labels))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
ax1.plot(Ks, inertias, marker='o'); ax1.set_title('Elbow (Inertia)'); ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
ax2.plot(Ks, sils, marker='o', color='green'); ax2.set_title('Silhouette vs K'); ax2.set_xlabel('K'); ax2.set_ylabel('Score')
plt.tight_layout(); plt.savefig('elbow_silhouette.png', dpi=150)
print("Saved elbow_silhouette.png")

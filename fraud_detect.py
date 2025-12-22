# fraud_detect.py
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# --- 1) Load dataset ---
fd = pd.read_csv("dataset/luxury_cosmetics_fraud_analysis_2025.csv")

# --- 2) Feature engineering & IQR Outlier Handling ---
time_objs = pd.to_datetime(fd['Transaction_Time'], format='%H:%M:%S')
fd['Time_Continuous'] = time_objs.dt.hour + time_objs.dt.minute / 60.0
fd['Day_of_Week'] = pd.to_datetime(fd['Transaction_Date']).dt.dayofweek

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df[column] = np.clip(df[column], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df

fd = cap_outliers_iqr(fd, 'Purchase_Amount')
fd = cap_outliers_iqr(fd, 'Customer_Age')

# --- 3) Preprocessing Pipeline ---
EXPECTED_FEATURES = [
    'Purchase_Amount', 'Customer_Age', 'Footfall_Count',
    'Time_Continuous', 'Day_of_Week',
    'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
]

num_cols = ['Purchase_Amount', 'Customer_Age', 'Footfall_Count', 'Time_Continuous', 'Day_of_Week']
cat_cols = ['Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', MinMaxScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
])

# --- 4) Balancing and Modeling ---
X_processed = preprocessor.fit_transform(fd[EXPECTED_FEATURES])
y = fd['Fraud_Flag']

smote = SMOTE(random_state=42)
X_resampled, _ = smote.fit_resample(X_processed, y)

kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(X_resampled)

# --- 5) Save Artifact ---
model_pipeline = Pipeline([('preprocess', preprocessor), ('cluster_model', kmeans)])
artifact = {'model': model_pipeline, 'expected_features': EXPECTED_FEATURES}

with open('fraud_detection_Final_model.pkl', 'wb') as f:
    pickle.dump(artifact, f)
print("✅ Mastery Model Saved!")

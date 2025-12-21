
# Payment Transaction Fraud – Unsupervised K-Means (Streamlit)

This repository contains:

- `train_model.py`: trains a K-Means clustering pipeline with feature engineering and auto-selects K via silhouette score; saves artifacts with **joblib** and **pickle**.
- `app.py`: Streamlit GUI for single-transaction clustering (unsupervised). Uses cached model loading and friendly diagnostics.
- `dev_notebook.py`: quick EDA and elbow/silhouette plot generator.
- `requirements.txt`: pinned versions for deployment consistency.
- `dataset/` (you add the CSV): place `luxury_cosmetics_fraud_analysis_2025.csv` here.

## How to run locally

```bash
# 1) Create/activate venv (optional)
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train the model (make sure dataset CSV exists)
python train_model.py

# 4) Run Streamlit app
streamlit run app.py
```

## Deployment tips
- Commit the generated `fraud_detection_model.joblib` (and `.pkl`) to the repo.
- On Streamlit Cloud, after pushing changes: **Manage app → Clear cache** → reload.
- Keep your `requirements.txt` pinned to avoid API differences.

## Notes
- Clustering is **unsupervised**; labels (if present) are **not used in training**—they can be used post-hoc for interpretation.
- The app expects the following feature schema:
  - Numeric: `Purchase_Amount`, `Customer_Age`, `Footfall_Count`, `Time_Continuous`, `Day_of_Week`
  - Categorical: `Customer_Loyalty_Tier`, `Payment_Method`, `Product_Category`


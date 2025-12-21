
# app.py
# Streamlit app to load the trained K-Means pipeline and predict cluster for a single transaction.

import os
import joblib
import pickle
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Fraud Detector", layout="wide")

st.sidebar.title("🛡️ Fraud Sentinel")
st.sidebar.image(
    "https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png",
    use_container_width=True
)
st.sidebar.markdown("---")
st.sidebar.subheader("Owner's Information")
st.sidebar.write("**Name:** Min Thant Hein")
st.sidebar.write("**ID:** PIUS20230001")

@st.cache_resource
def load_artifact():
    # Prefer joblib; fallback to pickle
    if os.path.exists("fraud_detection_model.joblib"):
        return joblib.load("fraud_detection_model.joblib")
    with open("fraud_detection_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_artifact()

# Handle both: dict artifact or direct Pipeline
if isinstance(artifact, dict):
    model = artifact.get('model')
    EXPECTED_FEATURES = artifact.get('expected_features') or [
        'Purchase_Amount', 'Customer_Age', 'Footfall_Count',
        'Time_Continuous', 'Day_of_Week',
        'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
    ]
else:
    model = artifact
    EXPECTED_FEATURES = [
        'Purchase_Amount', 'Customer_Age', 'Footfall_Count',
        'Time_Continuous', 'Day_of_Week',
        'Customer_Loyalty_Tier', 'Payment_Method', 'Product_Category'
    ]

if not isinstance(model, Pipeline):
    st.error("Loaded object is not a scikit-learn Pipeline. Re-train and pickle correctly.")
    st.stop()

st.title("🛡️ Luxury Cosmetics Fraud Detection (Unsupervised K-Means)")
st.write("Enter transaction details to identify the behavioral cluster.\n"
         "Note: Clustering is unsupervised; labels are not used in training.")

col1, col2, col3 = st.columns(3)
with col1:
    amt = st.number_input("Purchase Amount ($)", value=500.0, step=10.0)
    loyalty = st.selectbox("Loyalty Tier", ["Gold", "Silver", "Bronze", "None"])
with col2:
    age = st.number_input("Customer Age", value=30, step=1)
    pay = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash", "Crypto"])
with col3:
    foot = st.number_input("Store Footfall", value=50, step=1)
    cat = st.selectbox("Product Category", ["Skincare", "Fragrance", "Makeup"])

hour = st.slider("Transaction Hour", 0, 23, 14)
day = st.selectbox("Day of Week", list(range(7)),
                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

if st.button("✨ Identify Transaction Cluster", type="primary"):
    input_df = pd.DataFrame([{
        'Purchase_Amount': float(amt),
        'Customer_Age': float(age),
        'Footfall_Count': float(foot),
        'Time_Continuous': float(hour),
        'Day_of_Week': int(day),
        'Customer_Loyalty_Tier': loyalty,
        'Payment_Method': pay,
        'Product_Category': cat
    }], columns=EXPECTED_FEATURES)

    try:
        result = int(model.predict(input_df)[0])
        st.markdown("---")
        st.success(f"### ✅ Transaction Identified: Cluster {result}")
        if result == 0:
            st.info("💡 **Insight:** Typical high-value customer segment.")
        else:
            st.warning("⚠️ **Insight:** Behavior aligns with a segment often flagged for review.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        try:
            pre = model.named_steps.get('preprocess')
            if pre is not None and hasattr(pre, 'transformers_'):
                num_cols = pre.transformers_[0][2] if len(pre.transformers_)>0 else []
                cat_cols = pre.transformers_[1][2] if len(pre.transformers_)>1 else []
                st.caption(f"Numeric expected by pipeline: {list(num_cols)}")
                st.caption(f"Categorical expected by pipeline: {list(cat_cols)}")
        except Exception:
            pass
        st.caption("Tip: Manage app → Clear cache after updating model/pickle.")

#fraud_detect2.py


# app.py
import pickle
import pandas as pd
import streamlit as st

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
st.sidebar.write("**Course:** Introduction to Machine Learning")
st.sidebar.write("**Professor:** Dr. Nwe Nwe Htay Win")
st.sidebar.markdown("---")


@st.cache_resource
def load_model_artifact():
    with open("fraud_detection_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_model_artifact()
model = artifact['model']
EXPECTED_FEATURES = artifact['expected_features']

st.title("🛡️ Luxury Cosmetics Fraud Detection (Unsupervised K-Means)")
st.write("Enter transaction details to identify the behavioral cluster.\n"
         "Note: Clustering is unsupervised; labels are not used in training.")

# UI
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
day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

# --- 4. Prediction Logic ---
if st.button("✨ Identify Transaction Cluster", type="primary"):
    input_df = pd.DataFrame({
        'Purchase_Amount': [float(amt)],
        'Customer_Age': [float(age)],
        'Footfall_Count': [float(foot)],
        'Time_Continuous': [float(hour)],
        'Day_of_Week': [day],
        'Customer_Loyalty_Tier': [loyalty],
        'Payment_Method': [pay],
        'Product_Category': [cat]
    })

    # Ensure column order matches training EXACTLY
    input_df = input_df[artifact['expected_features']]
    
    # Predict
    cluster_id = artifact['model'].predict(input_df)[0]

    # --- NEW: Cluster Interpretation Mapping ---
    cluster_map = {
        0: "Standard High-Value Buyers",
        1: "Suspicious Late-Night Anomalies",
        2: "Typical Retail Customers",
        3: "New Account/High-Velocity Risks",
        4: "Verified VIP Segment"
    }
    persona = cluster_map.get(cluster_id, "Unknown Segment")

    st.markdown("---")
    st.success(f"### ✅ Result: {persona} (Cluster {cluster_id})")
    
    # Recommendation Hook
    if cluster_id in [1, 3]:
        st.warning("⚠️ **Action:** Flag for manual security review.")
    else:
        st.info("💡 **Action:** Standard processing recommended.")



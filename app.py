# app.py
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Sentinel", layout="wide")

st.sidebar.title("🛡️ Fraud Sentinel")
st.sidebar.image("https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Owner's Information")
st.sidebar.write("**Name:** Min Thant Hein")
st.sidebar.write("**ID:** PIUS20230001")
st.sidebar.write("**Course:** Introduction to Machine Learning")
st.sidebar.write("**Professor:** Dr. Nwe Nwe Htay Win")
st.sidebar.markdown("---")

@st.cache_resource
def load_model():
    with open("fraud_detection_Final_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_model()
model = artifact['model']
EXPECTED_FEATURES = artifact['expected_features']

st.title("🛡️ Luxury Cosmetics Fraud Detection")
st.write("Using Unsupervised K-Means ($K=4$) to detect behavioral anomalies.")

col1, col2, col3 = st.columns(3)
with col1:
    amt = st.number_input("Purchase Amount ($)", value=100.0)
    loyalty = st.selectbox("Loyalty Tier", ["Gold", "Silver", "Bronze", "None"])
with col2:
    age = st.number_input("Customer Age", value=25)
    pay = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash", "Crypto"])
with col3:
    foot = st.number_input("Store Footfall", value=50)
    cat = st.selectbox("Product Category", ["Skincare", "Fragrance", "Makeup"])

hour = st.slider("Transaction Hour", 0, 23, 14)
day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

if st.button("✨ Identify Cluster", type="primary"):
    # 1. Human Logic Filter (Mastery Point)
    if age > 100 or age < 12:
        st.error("🚨 ALERT: Physically impossible age detected. Manual fraud investigation required.")
    else:
        input_df = pd.DataFrame([{
            'Purchase_Amount': float(amt), 'Customer_Age': float(age), 'Footfall_Count': float(foot),
            'Time_Continuous': float(hour), 'Day_of_Week': int(day),
            'Customer_Loyalty_Tier': loyalty, 'Payment_Method': pay, 'Product_Category': cat
        }], columns=EXPECTED_FEATURES)

        cluster = int(model.predict(input_df)[0])
        
        # 2. Persona Mapping
        cluster_map = {
            0: {"name": "Standard Buyer", "color": "blue", "rec": "Auto-approve."},
            1: {"name": "High-Value Anomaly", "color": "red", "rec": "Block & Verify ID."},
            2: {"name": "Night-time Suspicious", "color": "orange", "rec": "Manual Review Required."},
            3: {"name": "Verified VIP", "color": "green", "rec": "Priority Shipping."}
        }
        
        res = cluster_map[cluster]
        st.markdown(f"### Result: :{res['color']}[{res['name']} (Cluster {cluster})]")
        st.info(f"💡 **Recommendation:** {res['rec']}")


import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Sentinel", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🛡️ Fraud Sentinel")
st.sidebar.image("https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.subheader("Project Details")
st.sidebar.write("**Student:** Min Thant Hein")
st.sidebar.write("**K-Value:** 4 (Optimized via Elbow & Silhouette)")
st.sidebar.write("**Method:** KMeans + MinMaxScaler + SMOTE")

# --- MAIN UI ---
st.title("🛡️ Luxury Cosmetics Fraud Detection")
st.write("Professional Decision Support System for Loss Prevention Teams.")

# --- NEW SECTION: DOS AND DON'TS (USER GUIDE) ---
with st.expander("ℹ️ User Guide: Dos and Don'ts for Fraud Detection"):
    col_do, col_dont = st.columns(2)
    with col_do:
        st.success("### ✅ Dos")
        st.write("- **Check High-Value Clusters:** Prioritize manual reviews for Cluster 1.")
        st.write("- **Verify Night Transactions:** Use the slider to check risk at late hours.")
        st.write("- **Validate VIPs:** Reward loyal customers in Cluster 3 with fast-track shipping.")
    with col_dont:
        st.error("### ❌ Don'ts")
        st.write("- **Ignore Age Alerts:** Never approve transactions with impossible ages.")
        st.write("- **Blindly Trust 0:** Even 'Standard' clusters need periodic audit.")
        st.write("- **Input Raw Data:** Ensure currency is converted to USD before entry.")

st.markdown("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure this filename matches your latest saved pickle
    with open("fraud_detection_Final_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_model()
model = artifact['model']
EXPECTED_FEATURES = artifact['expected_features']

# --- INPUT FORM ---
col1, col2, col3 = st.columns(3)
with col1:
    amt = st.number_input("Purchase Amount ($)", value=100.0, help="Total transaction value in USD")
    loyalty = st.selectbox("Loyalty Tier", ["Gold", "Silver", "Bronze", "None"])
with col2:
    age = st.number_input("Customer Age", value=25, min_value=1, max_value=120)
    pay = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Cash", "Crypto"])
with col3:
    foot = st.number_input("Store Footfall", value=50)
    cat = st.selectbox("Product Category", ["Skincare", "Fragrance", "Makeup"])

hour = st.slider("Transaction Hour", 0, 23, 14)
day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

# --- PREDICTION & SPECIFIC RECOMMENDATIONS ---
if st.button("✨ Analyze Transaction", type="primary"):
    input_df = pd.DataFrame([{
        'Purchase_Amount': float(amt), 'Customer_Age': float(age), 'Footfall_Count': float(foot),
        'Time_Continuous': float(hour), 'Day_of_Week': int(day),
        'Customer_Loyalty_Tier': loyalty, 'Payment_Method': pay, 'Product_Category': cat
    }], columns=EXPECTED_FEATURES)

    cluster = int(model.predict(input_df)[0])
    
    # Updated logic for Cluster Personas and Specific Recommendations
    cluster_info = {
        0: {
            "name": "Standard Everyday Buyer",
            "color": "blue",
            "rec": "✅ **Action: Auto-Approve.** This profile fits typical shopping patterns. No further action needed.",
            "risk": "Low"
        },
        1: {
            "name": "High-Value Alert",
            "color": "red",
            "rec": "🚨 **Action: IMMEDIATE HOLD.** Large amount outlier. Contact customer via phone to verify identity before shipping.",
            "risk": "High"
        },
        2: {
            "name": "Late-Night Anomaly",
            "color": "orange",
            "rec": "⚠️ **Action: Manual Review.** Transaction occurred at an unusual hour. Cross-reference IP address with shipping address.",
            "risk": "Medium"
        },
        3: {
            "name": "Verified VIP Client",
            "color": "green",
            "rec": "💎 **Action: Priority Handling.** Trusted loyalty member. Approve and apply complimentary express shipping.",
            "risk": "Safe"
        }
    }
    
    res = cluster_info.get(cluster, {"name": "Unknown", "color": "white", "rec": "Check logs.", "risk": "N/A"})
    
    st.markdown(f"### Result: :{res['color']}[{res['name']} (Cluster {cluster})]")
    st.write(f"**Calculated Risk Level:** {res['risk']}")
    st.info(res['rec'])

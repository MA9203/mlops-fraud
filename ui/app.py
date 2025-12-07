import streamlit as st
import requests
import pandas as pd

# Add a simple health check endpoint
# This would typically be handled by a separate lightweight server
# For now, we'll just rely on the container starting successfully

API_URL = "http://api:8000"

# ============================================================
# 🔹 App UI
# ============================================================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("💳 Fraud Detection Dashboard")

# ============================================================
# 🔹 Sidebar menu
# ============================================================
menu = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Monitoring"]
)

# ============================================================
# 🔹 Prediction Page
# ============================================================
if menu == "Prediction":

    st.header("🔍 Make a Prediction")

    st.markdown("Provide the 30 PCA features:")

    features = []
    cols = st.columns(3)

    for i in range(30):
        with cols[i % 3]:
            val = st.number_input(f"V{i+1}", value=0.0, step=0.01)
            features.append(val)

    if st.button("Predict 🚀"):
        payload = {"features": features}

        try:
            res = requests.post(f"{API_URL}/predict", json=payload).json()

            if "error" in res:
                st.error("❌ " + res["error"])
            else:
                st.success(f"Fraud Prediction : {res['fraud_prediction']}")
                st.write(f"Probability : **{res['fraud_probability']:.4f}**")

        except:
            st.error("❌ API unreachable.")

# ============================================================
# 🔹 Monitoring Page
# ============================================================
if menu == "Monitoring":

    st.header("📊 Monitoring & Analytics")

    # ---- Request metrics ----
    try:
        data = requests.get(f"{API_URL}/metrics").json()
    except:
        st.error("❌ Unable to contact API metrics endpoint.")
        st.stop()

    if "error" in data:
        st.error("❌ " + data["error"])
        st.stop()

    # ---- Convert dict to DataFrame ----
    df = pd.DataFrame([data])

    st.subheader("📄 Raw Monitoring Data")
    st.dataframe(df)

    # -------------------------
    # 📌 Basic KPIs
    # -------------------------
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", int(df["total_predictions"][0]))

    with col2:
        st.metric("Fraud Predictions", int(df["fraud_predictions"][0]))

    with col3:
        st.metric("Fraud Rate", f"{df['fraud_rate'][0] * 100:.2f}%")

    with col4:
        st.metric("Avg Probability", f"{df['average_probability'][0]:.6f}")

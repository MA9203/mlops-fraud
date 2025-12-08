import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Use localhost when running locally, api when running in Docker
API_URL = os.getenv("API_URL", "http://localhost:8000")

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
    ["Prediction", "Monitoring", "Data Drift", "Model Drift"]
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
                st.success(f"Fraud Prediction : {res['prediction']}")
                st.write(f"Probability : **{res['probability']:.4f}**")

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

    # -------------------------
    # 📈 Drift Visualization Over Time
    # -------------------------
    st.subheader("📈 Drift Visualization Over Time")
    
    # Try to load historical drift data
    try:
        # Check if we have data drift reports
        if os.path.exists("reports/data_drift_report.json"):
            with open("reports/data_drift_report.json", "r") as f:
                drift_data = json.load(f)
            
            # Create a simple visualization of drift over time
            st.info("Historical drift data visualization would appear here in a production environment.")
            st.write("In a full implementation, this would show:")
            st.write("- Feature drift trends over time")
            st.write("- Statistical significance of drift")
            st.write("- Alert thresholds")
        else:
            st.info("No historical drift data available yet. Run drift analysis to generate reports.")
    except Exception as e:
        st.warning(f"Could not load historical drift data: {str(e)}")

# ============================================================
# 🔹 Data Drift Monitoring Page
# ============================================================
if menu == "Data Drift":
    st.header("🔬 Data Drift Monitoring")
    
    st.markdown("""
    This section monitors data drift between the training data and the current production data.
    Data drift can indicate when the model may need retraining.
    """)
    
    # Button to trigger drift detection
    if st.button("Run Data Drift Analysis 🔄"):
        try:
            with st.spinner("Running data drift analysis..."):
                # Trigger drift detection
                response = requests.post(f"{API_URL}/monitoring/drift/check").json()
                
            st.success("Data drift analysis completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Drift Detected", "Yes" if response["drift_detected"] else "No")
            
            with col2:
                st.metric("Drifted Features", response["number_of_drifted_features"])
            
            with col3:
                st.metric("Total Features", response["total_features"])
            
            # Show detailed report
            st.subheader("Detailed Report")
            try:
                report = requests.get(f"{API_URL}/monitoring/drift/report").json()
                st.json(report)
            except:
                st.info("Detailed report not available")
                
        except Exception as e:
            st.error(f"Failed to run data drift analysis: {str(e)}")
    
    # Show latest drift report if available
    st.subheader("Latest Drift Report")
    try:
        report = requests.get(f"{API_URL}/monitoring/drift/report").json()
        
        # Extract drift details for visualization
        if "data_drift" in report and "features" in report["data_drift"]:
            features_data = report["data_drift"]["features"]
            
            # Create DataFrame for visualization
            features_df = pd.DataFrame([
                {
                    "Feature": feature,
                    "KS Statistic": details.get("ks_statistic", 0),
                    "P-Value": details.get("p_value", 1),
                    "Drift Detected": details.get("drift_detected", False)
                }
                for feature, details in features_data.items()
                if "error" not in details
            ])
            
            if not features_df.empty:
                # Heatmap-like visualization using bar chart
                st.subheader("📊 Feature Drift Visualization")
                
                # Color coding for drift detection
                features_df["Color"] = features_df["Drift Detected"].map({True: "Drift Detected", False: "No Drift"})
                
                # Bar chart showing KS statistics
                fig_ks = px.bar(
                    features_df, 
                    x="Feature", 
                    y="KS Statistic", 
                    color="Color",
                    title="KS Statistic by Feature (Higher values indicate more drift)",
                    color_discrete_map={"Drift Detected": "red", "No Drift": "green"}
                )
                st.plotly_chart(fig_ks, use_container_width=True)
                
                # Bar chart showing P-values
                fig_p = px.bar(
                    features_df, 
                    x="Feature", 
                    y="P-Value", 
                    color="Color",
                    title="P-Values by Feature (Values below threshold indicate drift)",
                    color_discrete_map={"Drift Detected": "red", "No Drift": "green"}
                )
                st.plotly_chart(fig_p, use_container_width=True)
                
                # Show the raw data
                st.subheader("📋 Detailed Drift Data")
                st.dataframe(features_df)
            else:
                st.info("No feature drift data available for visualization")
        else:
            st.json(report)
    except Exception as e:
        st.info("No drift report available. Run analysis first.")

# ============================================================
# 🔹 Model Drift Monitoring Page
# ============================================================
if menu == "Model Drift":
    st.header("🤖 Model Drift Monitoring")
    
    st.markdown("""
    This section monitors model drift by comparing current predictions with reference metrics.
    Model drift can indicate when the model's performance is degrading.
    """)
    
    # Button to trigger model drift detection
    if st.button("Run Model Drift Analysis 🔄"):
        try:
            with st.spinner("Running model drift analysis..."):
                # Trigger model drift detection
                response = requests.post(f"{API_URL}/monitoring/model/check").json()
                
            st.success("Model drift analysis completed!")
            
            # Display results
            st.metric("Model Drift Detected", "Yes" if response["drift_detected"] else "No")
            
            # Show detailed metrics comparison
            st.subheader("Metrics Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Reference Metrics**")
                ref_metrics = response["reference_metrics"]
                for key, value in ref_metrics.items():
                    st.write(f"{key}: {value:.6f}")
            
            with col2:
                st.markdown("**Production Metrics**")
                prod_metrics = response["production_metrics"]
                for key, value in prod_metrics.items():
                    if isinstance(value, (int, float)):
                        st.write(f"{key}: {value:.6f}")
                    else:
                        st.write(f"{key}: {value}")
            
            # Show drift details
            st.subheader("Drift Details")
            drift_details = response["drift_details"]
            for metric, details in drift_details.items():
                if details.get("drift_detected", False):
                    st.warning(f"⚠️ {metric}: Drift detected")
                    if "difference" in details:
                        st.write(f"  Difference: {details['difference']:.6f}")
                    if "change_ratio" in details:
                        st.write(f"  Change ratio: {details['change_ratio']:.6f}")
                else:
                    st.success(f"✅ {metric}: No drift detected")
                    
        except Exception as e:
            st.error(f"Failed to run model drift analysis: {str(e)}")
    
    # Show latest model drift report if available
    st.subheader("Latest Model Drift Report")
    try:
        report = requests.get(f"{API_URL}/monitoring/model/report").json()
        
        # Create visualization of model metrics over time
        st.subheader("📈 Model Performance Trends")
        
        # Extract metrics for visualization
        if "production_metrics" in report:
            prod_metrics = report["production_metrics"]
            
            # Create metrics dataframe
            metrics_data = []
            for key, value in prod_metrics.items():
                if isinstance(value, (int, float)) and key not in ["total_predictions"]:
                    metrics_data.append({"Metric": key, "Value": value})
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                # Bar chart of current metrics
                fig_metrics = px.bar(
                    metrics_df,
                    x="Metric",
                    y="Value",
                    title="Current Model Performance Metrics"
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Show detailed report
            st.json(report)
        else:
            st.json(report)
    except:
        st.info("No model drift report available. Run analysis first.")
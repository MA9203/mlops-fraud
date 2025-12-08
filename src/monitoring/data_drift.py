import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import warnings
warnings.filterwarnings('ignore')

# Configuration
REFERENCE_DATA_PATH = "data/processed/creditcard_processed.csv"
MONITORING_LOG_PATH = "monitoring/predictions_log.json"
REPORTS_DIR = "reports"
DATA_DRIFT_REPORT_HTML = os.path.join(REPORTS_DIR, "data_drift_report.html")
DATA_DRIFT_REPORT_JSON = os.path.join(REPORTS_DIR, "data_drift_report.json")

def load_reference_data():
    """Load the reference dataset used for training"""
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Reference data not found at {REFERENCE_DATA_PATH}")
    
    df = pd.read_csv(REFERENCE_DATA_PATH)
    # Drop the target column for drift analysis
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)
    return df

def load_current_data():
    """Load current data from prediction logs"""
    if not os.path.exists(MONITORING_LOG_PATH):
        raise FileNotFoundError(f"Monitoring log not found at {MONITORING_LOG_PATH}")
    
    with open(MONITORING_LOG_PATH, 'r') as f:
        logs = json.load(f)
    
    # Extract features from logs
    features_list = [entry['features'] for entry in logs]
    
    # Get feature names from reference data
    reference_df = load_reference_data()
    feature_names = reference_df.columns.tolist()
    
    # Create DataFrame with current data
    current_df = pd.DataFrame(features_list, columns=feature_names)
    return current_df

def detect_feature_drift(reference_data, current_data):
    """Detect feature drift between reference and current data"""
    # Define column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = reference_data.columns.tolist()
    
    # Create data drift profile
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference_data, current_data, column_mapping=column_mapping)
    
    # Get results
    drift_results = data_drift_profile.json()
    return json.loads(drift_results)

def generate_data_drift_dashboard(reference_data, current_data, output_path):
    """Generate HTML dashboard for data drift visualization"""
    # Define column mapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = reference_data.columns.tolist()
    
    # Create dashboard
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(reference_data, current_data, column_mapping=column_mapping)
    dashboard.save(output_path)
    
    print(f"Data drift dashboard saved to {output_path}")

def save_drift_report(report_data, output_path):
    """Save drift report as JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Data drift report saved to {output_path}")

def run_data_drift_monitoring():
    """Main function to run data drift monitoring"""
    print("Starting data drift monitoring...")
    
    try:
        # Load data
        print("Loading reference data...")
        reference_data = load_reference_data()
        
        print("Loading current data...")
        current_data = load_current_data()
        
        # Ensure both datasets have the same columns
        common_columns = list(set(reference_data.columns) & set(current_data.columns))
        reference_data = reference_data[common_columns]
        current_data = current_data[common_columns]
        
        print(f"Reference data shape: {reference_data.shape}")
        print(f"Current data shape: {current_data.shape}")
        
        # Detect drift
        print("Detecting data drift...")
        drift_report = detect_feature_drift(reference_data, current_data)
        
        # Save JSON report
        os.makedirs(REPORTS_DIR, exist_ok=True)
        save_drift_report(drift_report, DATA_DRIFT_REPORT_JSON)
        
        # Generate HTML dashboard
        generate_data_drift_dashboard(reference_data, current_data, DATA_DRIFT_REPORT_HTML)
        
        # Print summary
        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_shape": reference_data.shape,
            "current_data_shape": current_data.shape,
            "drift_detected": drift_report.get("data_drift", {}).get("data_drift_detected", False),
            "number_of_drifted_features": drift_report.get("data_drift", {}).get("n_drifted_features", 0),
            "total_features": drift_report.get("data_drift", {}).get("n_features", 0)
        }
        
        print("\n=== Data Drift Monitoring Summary ===")
        print(f"Timestamp: {drift_summary['timestamp']}")
        print(f"Reference data: {drift_summary['reference_data_shape']}")
        print(f"Current data: {drift_summary['current_data_shape']}")
        print(f"Drift detected: {drift_summary['drift_detected']}")
        print(f"Drifted features: {drift_summary['number_of_drifted_features']}/{drift_summary['total_features']}")
        
        return drift_summary
        
    except Exception as e:
        print(f"Error during data drift monitoring: {str(e)}")
        raise

if __name__ == "__main__":
    run_data_drift_monitoring()
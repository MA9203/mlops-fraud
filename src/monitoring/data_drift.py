import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats
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

def ks_test_drift(reference_data, current_data, threshold=0.05):
    """Perform Kolmogorov-Smirnov test for drift detection"""
    drift_results = {}
    drifted_features = 0
    
    for column in reference_data.columns:
        if column in current_data.columns:
            # Perform KS test
            try:
                ks_statistic, p_value = stats.ks_2samp(
                    reference_data[column].dropna(), 
                    current_data[column].dropna()
                )
                
                # Drift detected if p-value < threshold
                drift_detected = p_value < threshold
                
                drift_results[column] = {
                    "ks_statistic": float(ks_statistic),
                    "p_value": float(p_value),
                    "drift_detected": drift_detected,
                    "threshold": threshold
                }
                
                if drift_detected:
                    drifted_features += 1
                    
            except Exception as e:
                drift_results[column] = {
                    "error": str(e),
                    "drift_detected": False
                }
    
    return drift_results, drifted_features

def detect_feature_drift(reference_data, current_data):
    """Detect feature drift between reference and current data"""
    # Perform KS test for drift detection
    drift_details, n_drifted_features = ks_test_drift(reference_data, current_data)
    
    # Overall drift detection (if any feature shows drift)
    dataset_drift = n_drifted_features > 0
    
    results = {
        "data_drift": {
            "dataset_drift": dataset_drift,
            "n_drifted_features": n_drifted_features,
            "n_features": len(reference_data.columns),
            "features": drift_details
        }
    }
    
    return results

def generate_simple_report(reference_data, current_data, output_path):
    """Generate a simple HTML report for data drift"""
    drift_results, _ = ks_test_drift(reference_data, current_data)
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Drift Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .drift { background-color: #ffcccc; }
            .no-drift { background-color: #ccffcc; }
        </style>
    </head>
    <body>
        <h1>Data Drift Report</h1>
        <h2>Summary</h2>
        <p>Generated on: {timestamp}</p>
        <table>
            <tr>
                <th>Feature</th>
                <th>KS Statistic</th>
                <th>P-Value</th>
                <th>Drift Detected</th>
            </tr>
    """.format(timestamp=datetime.now().isoformat())
    
    for feature, results in drift_results.items():
        if "error" not in results:
            css_class = "drift" if results["drift_detected"] else "no-drift"
            html_content += f"""
            <tr class="{css_class}">
                <td>{feature}</td>
                <td>{results['ks_statistic']:.4f}</td>
                <td>{results['p_value']:.4f}</td>
                <td>{'Yes' if results['drift_detected'] else 'No'}</td>
            </tr>
            """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Data drift report saved to {output_path}")

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
        generate_simple_report(reference_data, current_data, DATA_DRIFT_REPORT_HTML)
        
        # Extract summary information
        data_drift = drift_report.get("data_drift", {})
        drift_detected = data_drift.get("dataset_drift", False)
        n_drifted_features = data_drift.get("n_drifted_features", 0)
        n_features = data_drift.get("n_features", 0)
        
        # Print summary
        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_data_shape": reference_data.shape,
            "current_data_shape": current_data.shape,
            "drift_detected": drift_detected,
            "number_of_drifted_features": n_drifted_features,
            "total_features": n_features
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
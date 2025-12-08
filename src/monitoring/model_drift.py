import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
LOG_FILE = "monitoring/predictions_log.json"
METRICS_FILE = "metrics.json"
MODEL_DRIFT_REPORT_JSON = "reports/model_drift_report.json"
REFERENCE_ROC_AUC = 0.9552478595546265  # Default value, will be updated from metrics.json

def load_reference_metrics():
    """Load reference metrics from training evaluation"""
    if not os.path.exists(METRICS_FILE):
        print(f"Warning: Reference metrics file {METRICS_FILE} not found. Using default ROC-AUC.")
        return {"roc_auc": REFERENCE_ROC_AUC}
    
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def load_production_data():
    """Load production predictions data"""
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"Production log file {LOG_FILE} not found.")
    
    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    
    return data

def calculate_production_metrics(production_data, window_size=None):
    """Calculate metrics from production data"""
    if window_size:
        # Use only the last window_size predictions
        production_data = production_data[-window_size:]
    
    if len(production_data) == 0:
        return {
            "total_predictions": 0,
            "fraud_predictions": 0,
            "fraud_rate": 0,
            "average_probability": 0,
            "roc_auc": None
        }
    
    total = len(production_data)
    frauds = sum(1 for entry in production_data if entry["prediction"] == 1)
    avg_proba = sum(entry["probability"] for entry in production_data) / total
    
    # Calculate ROC-AUC if we have both predictions and probabilities
    # Note: In real production, we might not have true labels
    # This is a simulation using logged data
    try:
        # For demonstration purposes, we'll simulate having true labels
        # In a real scenario, you'd need to collect actual outcomes
        # This assumes we're comparing recent predictions with historical patterns
        probabilities = [entry["probability"] for entry in production_data]
        predictions = [entry["prediction"] for entry in production_data]
        
        # Simple anomaly detection based on probability distributions
        prob_mean = np.mean(probabilities)
        prob_std = np.std(probabilities)
        
        # Check for abnormal probabilities (more than 2 std away from mean)
        abnormal_probs = sum(1 for p in probabilities if abs(p - prob_mean) > 2 * prob_std)
        abnormal_rate = abnormal_probs / total if total > 0 else 0
        
        metrics = {
            "total_predictions": total,
            "fraud_predictions": frauds,
            "fraud_rate": frauds / total if total > 0 else 0,
            "average_probability": avg_proba,
            "probability_std": prob_std,
            "abnormal_probability_rate": abnormal_rate,
            "prob_mean": prob_mean
        }
        
    except Exception as e:
        print(f"Warning: Could not calculate detailed metrics: {e}")
        metrics = {
            "total_predictions": total,
            "fraud_predictions": frauds,
            "fraud_rate": frauds / total if total > 0 else 0,
            "average_probability": avg_proba,
            "probability_std": 0,
            "abnormal_probability_rate": 0,
            "prob_mean": avg_proba
        }
    
    return metrics

def detect_model_drift(reference_metrics, production_metrics, thresholds=None):
    """Detect model drift based on metric changes"""
    if thresholds is None:
        thresholds = {
            "roc_auc_drop": 0.05,  # 5% drop in ROC-AUC
            "fraud_rate_change": 0.5,  # 50% change in fraud rate
            "avg_probability_change": 0.1,  # 10% change in average probability
            "abnormal_probability_rate": 0.1  # 10% abnormal probabilities
        }
    
    drift_detected = False
    drift_details = {}
    
    # Compare ROC-AUC (if available)
    ref_roc = reference_metrics.get("roc_auc", REFERENCE_ROC_AUC)
    prod_roc = production_metrics.get("roc_auc", ref_roc)
    
    if prod_roc is not None:
        roc_drop = ref_roc - prod_roc
        drift_details["roc_auc"] = {
            "reference": ref_roc,
            "production": prod_roc,
            "difference": roc_drop,
            "drift_detected": roc_drop > thresholds["roc_auc_drop"]
        }
        if roc_drop > thresholds["roc_auc_drop"]:
            drift_detected = True
    
    # Compare fraud rate
    ref_fraud_rate = reference_metrics.get("fraud_rate", 0)
    prod_fraud_rate = production_metrics.get("fraud_rate", 0)
    
    if ref_fraud_rate > 0:
        fraud_rate_change = abs(prod_fraud_rate - ref_fraud_rate) / ref_fraud_rate
    else:
        fraud_rate_change = abs(prod_fraud_rate - ref_fraud_rate)
    
    drift_details["fraud_rate"] = {
        "reference": ref_fraud_rate,
        "production": prod_fraud_rate,
        "change_ratio": fraud_rate_change,
        "drift_detected": fraud_rate_change > thresholds["fraud_rate_change"]
    }
    if fraud_rate_change > thresholds["fraud_rate_change"]:
        drift_detected = True
    
    # Compare average probability
    ref_avg_prob = reference_metrics.get("average_probability", 0)
    prod_avg_prob = production_metrics.get("average_probability", 0)
    
    if ref_avg_prob > 0:
        avg_prob_change = abs(prod_avg_prob - ref_avg_prob) / ref_avg_prob
    else:
        avg_prob_change = abs(prod_avg_prob - ref_avg_prob)
    
    drift_details["average_probability"] = {
        "reference": ref_avg_prob,
        "production": prod_avg_prob,
        "change_ratio": avg_prob_change,
        "drift_detected": avg_prob_change > thresholds["avg_probability_change"]
    }
    if avg_prob_change > thresholds["avg_probability_change"]:
        drift_detected = True
    
    # Check for abnormal probabilities
    abnormal_rate = production_metrics.get("abnormal_probability_rate", 0)
    drift_details["abnormal_probabilities"] = {
        "rate": abnormal_rate,
        "threshold": thresholds["abnormal_probability_rate"],
        "drift_detected": abnormal_rate > thresholds["abnormal_probability_rate"]
    }
    if abnormal_rate > thresholds["abnormal_probability_rate"]:
        drift_detected = True
    
    return drift_detected, drift_details

def generate_model_drift_report(reference_metrics, production_metrics, drift_detected, drift_details):
    """Generate a comprehensive model drift report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_metrics": reference_metrics,
        "production_metrics": production_metrics,
        "drift_detected": drift_detected,
        "drift_details": drift_details,
        "summary": {
            "total_metrics_compared": len(drift_details),
            "metrics_with_drift": sum(1 for detail in drift_details.values() if detail.get("drift_detected", False)),
        }
    }
    
    return report

def save_model_drift_report(report, output_path):
    """Save model drift report as JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Model drift report saved to {output_path}")

def run_model_drift_monitoring(window_size=1000):
    """Main function to run model drift monitoring"""
    print("Starting model drift monitoring...")
    
    try:
        # Load reference metrics
        print("Loading reference metrics...")
        reference_metrics = load_reference_metrics()
        
        # Load production data
        print("Loading production data...")
        production_data = load_production_data()
        
        # Calculate production metrics
        print("Calculating production metrics...")
        production_metrics = calculate_production_metrics(production_data, window_size)
        
        # Detect drift
        print("Detecting model drift...")
        drift_detected, drift_details = detect_model_drift(reference_metrics, production_metrics)
        
        # Generate report
        report = generate_model_drift_report(reference_metrics, production_metrics, drift_detected, drift_details)
        
        # Save report
        save_model_drift_report(report, MODEL_DRIFT_REPORT_JSON)
        
        # Print summary
        print("\n=== Model Drift Monitoring Summary ===")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total predictions analyzed: {production_metrics['total_predictions']}")
        print(f"Model drift detected: {drift_detected}")
        print(f"Metrics with drift: {report['summary']['metrics_with_drift']}/{report['summary']['total_metrics_compared']}")
        
        if drift_detected:
            print("\n⚠️  MODEL DRIFT DETECTED!")
            for metric, details in drift_details.items():
                if details.get("drift_detected", False):
                    print(f"  - {metric}: {details}")
        else:
            print("\n✅ No significant model drift detected.")
        
        return report
        
    except Exception as e:
        print(f"Error during model drift monitoring: {str(e)}")
        raise

if __name__ == "__main__":
    run_model_drift_monitoring()
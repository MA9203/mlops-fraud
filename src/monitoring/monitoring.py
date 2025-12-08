import os
import json
import datetime

# Configuration for request logging
REQUESTS_LOG_DIR = "logs/requests/"

# =========================================
# 1. Logging des prédictions
# =========================================
def log_prediction(features, prediction, probability):
    """Log prediction requests to individual files in logs/requests/ directory"""
    
    try:
        # Create logs directory structure
        os.makedirs(REQUESTS_LOG_DIR, exist_ok=True)
        
        # Create timestamp-based filename
        timestamp = datetime.datetime.now()
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = os.path.join(REQUESTS_LOG_DIR, filename)
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "features": features,
            "prediction": prediction,
            "probability": probability
        }
        
        # Write to individual file
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=4)
        
        # Also maintain the existing aggregated log for backward compatibility
        LOG_FILE = "monitoring/predictions_log.json"
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump([log_entry], f, indent=4)
        else:
            try:
                with open(LOG_FILE, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
            
            data.append(log_entry)
            
            with open(LOG_FILE, "w") as f:
                json.dump(data, f, indent=4)
                
    except Exception as e:
        # Log error but don't fail the prediction
        print(f"Warning: Failed to log prediction: {e}")


# =========================================
# 2. Génération de métriques globales
# =========================================
def generate_metrics():
    """Generate global metrics from prediction logs"""
    LOG_FILE = "monitoring/predictions_log.json"
    
    if not os.path.exists(LOG_FILE):
        return {"error": "No prediction logs found"}
    
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"error": "Could not read prediction logs"}
    
    total = len(data)
    frauds = sum(1 for entry in data if entry["prediction"] == 1)
    avg_proba = sum(entry["probability"] for entry in data) / total if total > 0 else 0
    
    return {
        "total_predictions": total,
        "fraud_predictions": frauds,
        "fraud_rate": frauds / total if total > 0 else 0,
        "average_probability": avg_proba,
    }


# =========================================
# 3. Utility functions for log management
# =========================================
def get_recent_requests(limit=100):
    """Get recent request logs for monitoring"""
    if not os.path.exists(REQUESTS_LOG_DIR):
        return []
    
    try:
        # Get all log files and sort by timestamp
        log_files = [f for f in os.listdir(REQUESTS_LOG_DIR) if f.endswith('.json')]
        log_files.sort(reverse=True)  # Most recent first
        
        recent_logs = []
        for filename in log_files[:limit]:
            try:
                with open(os.path.join(REQUESTS_LOG_DIR, filename), "r") as f:
                    log_entry = json.load(f)
                    recent_logs.append(log_entry)
            except Exception:
                continue  # Skip corrupted files
        
        return recent_logs
    except Exception:
        return []


def get_request_count():
    """Get total number of logged requests"""
    if not os.path.exists(REQUESTS_LOG_DIR):
        return 0
    
    try:
        return len([f for f in os.listdir(REQUESTS_LOG_DIR) if f.endswith('.json')])
    except Exception:
        return 0
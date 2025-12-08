#!/usr/bin/env python3
"""
Scheduled model drift monitoring script
This script can be run daily or triggered by new data
"""

import os
import sys
import argparse
from datetime import datetime
import json

# Add src to path to import monitoring modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.model_drift import run_model_drift_monitoring


def main():
    parser = argparse.ArgumentParser(description='Run scheduled model drift monitoring')
    parser.add_argument('--output-dir', default='reports', help='Directory to save reports')
    parser.add_argument('--window-size', type=int, default=1000, help='Number of recent predictions to analyze')
    parser.add_argument('--force', action='store_true', help='Force run even if no new data')
    
    args = parser.parse_args()
    
    print(f"Scheduled Model Drift Monitoring - {datetime.now().isoformat()}")
    print("=" * 50)
    
    try:
        # Run model drift monitoring
        model_drift_report = run_model_drift_monitoring(window_size=args.window_size)
        
        # Save summary to a timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(args.output_dir, f"model_drift_summary_{timestamp}.json")
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(model_drift_report, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        
        # Check if drift was detected
        if model_drift_report.get("drift_detected", False):
            print("\n⚠️  MODEL DRIFT DETECTED!")
            print(f"Number of metrics with drift: {model_drift_report['summary']['metrics_with_drift']}")
            print("Consider retraining the model or investigating prediction quality.")
            
            # Exit with error code to alert monitoring systems
            sys.exit(1)
        else:
            print("\n✅ No significant model drift detected.")
            sys.exit(0)
            
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("This might be expected if there's no prediction data yet.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
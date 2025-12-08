#!/usr/bin/env python3
"""
Scheduled monitoring script for data drift detection
This script can be run daily or triggered by new data
"""

import os
import sys
import argparse
from datetime import datetime
import json

# Add src to path to import monitoring modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.monitoring.data_drift import run_data_drift_monitoring


def main():
    parser = argparse.ArgumentParser(description='Run scheduled data drift monitoring')
    parser.add_argument('--output-dir', default='reports', help='Directory to save reports')
    parser.add_argument('--force', action='store_true', help='Force run even if no new data')
    
    args = parser.parse_args()
    
    print(f"Scheduled Data Drift Monitoring - {datetime.now().isoformat()}")
    print("=" * 50)
    
    try:
        # Run data drift monitoring
        drift_summary = run_data_drift_monitoring()
        
        # Save summary to a timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(args.output_dir, f"drift_summary_{timestamp}.json")
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        
        # Check if drift was detected
        if drift_summary.get("drift_detected", False):
            print("\n⚠️  DATA DRIFT DETECTED!")
            print(f"Number of drifted features: {drift_summary['number_of_drifted_features']}")
            print("Consider retraining the model or investigating the data source.")
            
            # Exit with error code to alert monitoring systems
            sys.exit(1)
        else:
            print("\n✅ No significant data drift detected.")
            sys.exit(0)
            
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("This might be expected if there's no data yet.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
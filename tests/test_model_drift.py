import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, 'src')

class TestModelDriftMonitoring(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.reports_dir = os.path.join(self.test_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('src.monitoring.model_drift.load_reference_metrics')
    @patch('src.monitoring.model_drift.load_production_data')
    def test_model_drift_imports(self, mock_production_data, mock_reference_metrics):
        """Test that model drift modules can be imported without errors."""
        try:
            from src.monitoring.model_drift import (
                load_reference_metrics,
                load_production_data,
                calculate_production_metrics,
                detect_model_drift,
                generate_model_drift_report,
                save_model_drift_report,
                run_model_drift_monitoring
            )
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_reference_metrics_loading(self):
        """Test loading of reference metrics."""
        from src.monitoring.model_drift import load_reference_metrics
        
        # Test with existing metrics file
        test_metrics = {"roc_auc": 0.95, "accuracy": 0.90}
        metrics_path = os.path.join(self.test_dir, 'test_metrics.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f)
        
        # Temporarily replace METRICS_FILE
        import src.monitoring.model_drift as model_drift_module
        original_metrics_file = model_drift_module.METRICS_FILE
        model_drift_module.METRICS_FILE = metrics_path
        
        try:
            metrics = load_reference_metrics()
            self.assertEqual(metrics["roc_auc"], 0.95)
        finally:
            model_drift_module.METRICS_FILE = original_metrics_file
    
    def test_drift_detection_logic(self):
        """Test drift detection logic."""
        from src.monitoring.model_drift import detect_model_drift
        
        # Test case with no drift
        reference_metrics = {"roc_auc": 0.95, "fraud_rate": 0.01, "average_probability": 0.05}
        production_metrics = {"roc_auc": 0.94, "fraud_rate": 0.011, "average_probability": 0.052}
        
        drift_detected, drift_details = detect_model_drift(reference_metrics, production_metrics)
        
        self.assertFalse(drift_detected, "No significant drift should be detected")
        
        # Test case with drift
        production_metrics_drift = {"roc_auc": 0.85, "fraud_rate": 0.05, "average_probability": 0.2}
        
        drift_detected, drift_details = detect_model_drift(reference_metrics, production_metrics_drift)
        
        self.assertTrue(drift_detected, "Significant drift should be detected")
    
    def test_report_generation_structure(self):
        """Test that model drift report has expected structure."""
        from src.monitoring.model_drift import generate_model_drift_report
        
        # Mock data
        reference_metrics = {"roc_auc": 0.95}
        production_metrics = {"roc_auc": 0.90}
        drift_detected = True
        drift_details = {
            "roc_auc": {
                "reference": 0.95,
                "production": 0.90,
                "difference": 0.05,
                "drift_detected": True
            }
        }
        
        # Generate report
        report = generate_model_drift_report(reference_metrics, production_metrics, drift_detected, drift_details)
        
        # Check report structure
        self.assertIn('timestamp', report)
        self.assertIn('reference_metrics', report)
        self.assertIn('production_metrics', report)
        self.assertIn('drift_detected', report)
        self.assertIn('drift_details', report)
        self.assertIn('summary', report)

if __name__ == '__main__':
    unittest.main()
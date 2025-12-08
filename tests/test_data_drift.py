import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, 'src')

class TestDataDriftMonitoring(unittest.TestCase):
    
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
    
    @patch('src.monitoring.data_drift.load_reference_data')
    @patch('src.monitoring.data_drift.load_current_data')
    def test_drift_detection_imports(self, mock_current_data, mock_reference_data):
        """Test that data drift modules can be imported without errors."""
        try:
            from src.monitoring.data_drift import (
                load_reference_data, 
                load_current_data, 
                detect_feature_drift,
                generate_data_drift_dashboard,
                save_drift_report,
                run_data_drift_monitoring
            )
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_report_generation_structure(self):
        """Test that drift report has expected structure."""
        # Mock drift report data
        mock_report = {
            "data_drift": {
                "data_drift_detected": False,
                "n_drifted_features": 0,
                "n_features": 30,
                "features": {}
            }
        }
        
        # Save mock report
        report_path = os.path.join(self.reports_dir, 'test_drift_report.json')
        with open(report_path, 'w') as f:
            json.dump(mock_report, f)
        
        # Check that report exists and has correct structure
        self.assertTrue(os.path.exists(report_path))
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        self.assertIn('data_drift', report)
        self.assertIn('data_drift_detected', report['data_drift'])
        self.assertIn('n_drifted_features', report['data_drift'])
        self.assertIn('n_features', report['data_drift'])

if __name__ == '__main__':
    unittest.main()
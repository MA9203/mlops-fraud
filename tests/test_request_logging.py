import unittest
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

class TestRequestLogging(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the REQUESTS_LOG_DIR and LOG_FILE paths
        import src.monitoring.monitoring as monitoring_module
        self.original_requests_log_dir = monitoring_module.REQUESTS_LOG_DIR
        self.original_log_file = getattr(monitoring_module, 'LOG_FILE', None)
        
        monitoring_module.REQUESTS_LOG_DIR = os.path.join(self.test_dir, 'logs', 'requests') + os.sep
        monitoring_module.LOG_FILE = os.path.join(self.test_dir, 'monitoring', 'predictions_log.json')
    
    def tearDown(self):
        """Clean up after each test method."""
        # Restore original paths
        import src.monitoring.monitoring as monitoring_module
        monitoring_module.REQUESTS_LOG_DIR = self.original_requests_log_dir
        if self.original_log_file:
            monitoring_module.LOG_FILE = self.original_log_file
            
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_log_prediction_creates_directories(self):
        """Test that log_prediction creates required directories."""
        from src.monitoring.monitoring import log_prediction
        
        # Test with sample data
        features = [0.1, 0.2, 0.3, 0.4, 0.5]
        prediction = 1
        probability = 0.85
        
        log_prediction(features, prediction, probability)
        
        # Check that directories were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'logs', 'requests')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'monitoring')))
        
    def test_log_prediction_creates_files(self):
        """Test that log_prediction creates individual log files."""
        from src.monitoring.monitoring import log_prediction, get_request_count
        
        # Test with sample data
        features = [0.1, 0.2, 0.3, 0.4, 0.5]
        prediction = 1
        probability = 0.85
        
        log_prediction(features, prediction, probability)
        
        # Check that individual log file was created
        request_count = get_request_count()
        self.assertGreaterEqual(request_count, 1)
        
        # Check that aggregated log file was created
        log_file_path = os.path.join(self.test_dir, 'monitoring', 'predictions_log.json')
        self.assertTrue(os.path.exists(log_file_path))
        
        # Check content of aggregated log
        with open(log_file_path, 'r') as f:
            data = json.load(f)
        
        self.assertGreaterEqual(len(data), 1)
        entry = data[-1]  # Get the last entry
        self.assertEqual(entry['features'], features)
        self.assertEqual(entry['prediction'], prediction)
        self.assertEqual(entry['probability'], probability)
        self.assertIn('timestamp', entry)
        
    def test_multiple_predictions_logged_correctly(self):
        """Test that multiple predictions are logged correctly."""
        from src.monitoring.monitoring import log_prediction, get_request_count, generate_metrics
        
        # Log multiple predictions
        test_data = [
            ([0.1, 0.2, 0.3], 0, 0.15),
            ([0.4, 0.5, 0.6], 1, 0.92),
            ([0.7, 0.8, 0.9], 0, 0.08),
        ]
        
        for features, prediction, probability in test_data:
            log_prediction(features, prediction, probability)
        
        # Check request count
        request_count = get_request_count()
        self.assertGreaterEqual(request_count, 3)
        
        # Check aggregated log
        log_file_path = os.path.join(self.test_dir, 'monitoring', 'predictions_log.json')
        self.assertTrue(os.path.exists(log_file_path))
        
        with open(log_file_path, 'r') as f:
            data = json.load(f)
        
        self.assertGreaterEqual(len(data), 3)
        
        # Check metrics
        metrics = generate_metrics()
        self.assertIn('total_predictions', metrics)
        self.assertGreaterEqual(metrics['total_predictions'], 3)
        
    def test_get_recent_requests(self):
        """Test that get_recent_requests returns correct data."""
        from src.monitoring.monitoring import log_prediction, get_recent_requests
        
        # Log a few predictions
        features = [0.1, 0.2, 0.3]
        log_prediction(features, 0, 0.15)
        log_prediction(features, 1, 0.92)
        
        # Get recent requests
        recent = get_recent_requests(limit=5)
        self.assertGreaterEqual(len(recent), 2)

if __name__ == '__main__':
    unittest.main()
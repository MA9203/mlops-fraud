# This file serves as the entry point for Railway deployment of the UI
# It runs the Streamlit application from ui/app.py

import subprocess
import sys
import os

if __name__ == "__main__":
    # Get port from environment variable or default to 8501
    port = os.environ.get("PORT", "8501")
    
    # Run the Streamlit app
    subprocess.run([
        "streamlit", "run", "ui/app.py", 
        "--server.port", port, 
        "--server.address", "0.0.0.0"
    ])
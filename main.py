# This file serves as the entry point for Railway deployment
# It imports and runs the FastAPI application from src/api.py

from src.api import app

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)
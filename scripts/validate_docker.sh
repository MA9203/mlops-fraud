#!/bin/bash

# Script to validate Docker images before publishing
set -e  # Exit on any error

echo "Starting Docker image validation..."

# Build all images
echo "Building Docker images..."
docker build -f docker/api.Dockerfile -t fraud-api-test .
docker build -f docker/train.Dockerfile -t fraud-train-test .
docker build -f docker/ui.Dockerfile -t fraud-ui-test .

# Test API image
echo "Testing API image..."
docker run -d --name test-api -p 8000:8000 fraud-api-test
sleep 10

# Check if container is running
if ! docker ps | grep -q test-api; then
  echo "ERROR: API container failed to start"
  docker logs test-api
  exit 1
fi

# Test health endpoint
echo "Checking API health endpoint..."
timeout 30 bash -c 'while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' localhost:8000/health)" != "200" ]]; do sleep 5; done' || {
  echo "ERROR: API health check failed"
  docker logs test-api
  exit 1
}

echo "API image validation passed!"
docker stop test-api

# Test UI image
echo "Testing UI image..."
docker run -d --name test-ui -p 8501:8501 fraud-ui-test
sleep 20

# Check if container is running
if ! docker ps | grep -q test-ui; then
  echo "ERROR: UI container failed to start"
  docker logs test-ui
  exit 1
fi

echo "UI image validation passed!"
docker stop test-ui

# Clean up
echo "Cleaning up test containers..."
docker rm test-api test-ui 2>/dev/null || true

echo "All Docker image validations passed!"
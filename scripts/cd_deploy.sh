#!/bin/bash

# Build Docker image
echo "Building Docker image..."
docker build -t fiveg_app:latest -f docker/Dockerfile.django .

# Tag Docker image with the latest commit SHA
echo "Tagging image with commit hash..."
docker tag fiveg_app:latest willisrunner/fiveg_app:latest

# Push Docker image to DockerHub
echo "Pushing image to DockerHub..."
docker push willisrunner/fiveg_app:latest

# Log model training in MLflow
echo "Logging model training in MLflow..."
python3 backend/ml/train.py

# Deploy with Docker Compose
echo "Deploying containers..."
docker-compose up -d

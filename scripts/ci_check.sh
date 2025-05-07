#!/bin/bash

# Run linting (flake8) and formatting (black)
echo "Running Black formatting..."
black backend/

echo "Running Flake8 for code linting..."
flake8 backend/

echo "Running Bandit for security checks..."
bandit -r backend/

echo "Running tests..."
pytest backend/  # Adjust if you have tests in specific directories

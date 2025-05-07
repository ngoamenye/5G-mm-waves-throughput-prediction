FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8
ENV PYTHON_VERSION=3.9

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install mlflow

# Copy application files
COPY . /app/

# Set the working directory
WORKDIR /app

# Run the app
CMD ["python", "train_tf_model.py"]

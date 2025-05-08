# Use an optimized Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# Install system dependencies (this step only happens if necessary)
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt first (this will be cached unless it changes)
COPY requirements.txt /app/

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (this will change more frequently)
COPY . /app/

# Expose the port your app will run on
EXPOSE 8000

# Set the default command
CMD ["python", "train_tf_model.py"]

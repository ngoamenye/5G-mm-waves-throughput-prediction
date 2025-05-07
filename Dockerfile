# Build stage
FROM python:3.9-slim as build-stage
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=build-stage /app /app
COPY . /app/
CMD ["python", "train_tf_model.py"]

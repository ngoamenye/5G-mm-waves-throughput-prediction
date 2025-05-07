#!/bin/bash

echo "✅ Initialisation du pipeline SynergyX..."

# Migrations Django
echo "📦 Migration de la base Django..."
python manage.py migrate

# Lancement Django
echo "🚀 Lancement de Django sur :8000"
python manage.py runserver 0.0.0.0:8000 &

# MLflow
echo "📊 MLflow UI sur :5000"
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 &

# Prometheus
echo "📈 Prometheus sur :9090"
prometheus --config.file=prometheus.yml &

# Grafana
echo "📊 Grafana sur :3000"
grafana-server --homepath=/usr/share/grafana &

# Node exporter (optionnel)
node_exporter &

wait

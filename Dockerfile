# Dockerfile
FROM python:3.9-slim

# Créer le dossier d’application
WORKDIR /app

# Copie du code et des fichiers de conf
COPY . /app

# Variables d’environnement pour Django
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installation des dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install mlflow

# Ports exposés
EXPOSE 8000 5000 9090

# Point d'entrée : on utilise un script pour lancer tous les services
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]

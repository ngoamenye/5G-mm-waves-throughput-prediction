from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import pandas as pd
import numpy as np
import joblib
import pickle
import psutil
import mlflow
from tensorflow import keras
from keras.models import load_model
import os
from django.views.decorators.http import require_GET



# Chargement du modèle et des outils
@require_GET
def map_prediction_api(request):
    try:
        # Charger le modèle et le scaler
        model = joblib.load("ml_models/throughput_model.keras")
        scaler = joblib.load("ml_models/scaler.gz")
        with open("ml_models/feature_names.pkl", "rb") as f:
            feature_names = joblib.load(f)

        # Extraire les paramètres GET
        lon = float(request.GET.get("lon", 0))
        lat = float(request.GET.get("lat", 0))
        speed = float(request.GET.get("speed", 0))
        direction = float(request.GET.get("direction", 0))
        ssRsrp = float(request.GET.get("nr_ssRsrp", -95))
        ssRsrq = float(request.GET.get("nr_ssRsrq", -10))
        ssSinr = float(request.GET.get("nr_ssSinr", 10))

        # Créer le vecteur d’entrée
        X = pd.DataFrame([{
            'longitude': lon,
            'latitude': lat,
            'speed': speed,
            'direction': direction,
            'nr_ssRsrp': ssRsrp,
            'nr_ssRsrq': ssRsrq,
            'nr_ssSinr': ssSinr
        }])

        X = X[feature_names]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0][0]

        return JsonResponse({"throughput": round(float(pred), 2)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def dashboard_view(request):
    # ✅ Exemple de métriques MLOps
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()
    experiments = client.list_experiments()

    # Exemple: dernière exécution
    runs = client.search_runs(experiments[0].experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    run = runs[0]

    # CPU et RAM (DataOps)
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    context = {
        "model_name": run.data.tags.get("mlflow.runName", "Non défini"),
        "accuracy": run.data.metrics.get("accuracy"),
        "loss": run.data.metrics.get("loss"),
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "grafana_url": "http://localhost:3000/d/synergyx_metrics",
        "mlflow_url": f"http://localhost:5000/#/experiments/{experiments[0].experiment_id}/runs/{run.info.run_id}"
    }

    return render(request, "simulator/dashboard.html", context)
@login_required
def dashboard_view(request):
    return render(request, 'dashboard.html')


def home_view(request):
    return render(request, 'home.html')








# Inscription
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'simulator/signup.html', {'form': form})

# Connexion
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)  # ✅ il faut passer request ici
        if form.is_valid():
            user = form.get_user()  # ✅ get_user() est sûr ici
            login(request, user)
            return redirect('dashboard')  # ✅ redirection vers tableau de bord
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})
# Déconnexion
def logout_view(request):
    logout(request)
    return redirect('login')

# Dashboard (protégé)
@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

# Page prédiction
@login_required
def predict_page(request):
    return render(request, 'predictor.html')

# Page drift
@login_required
def drift_page(request):
    return render(request, 'drift.html')

@login_required
def map_page(request):
    return render(request, 'simulator/map.html')
# Page map
def map_view(request):
    return render(request, 'map.html')

@csrf_exempt
@require_GET
def map_prediction(request):
    try:
        # Récupération des paramètres
        params = {k: float(request.GET.get(k, 0)) for k in feature_names}

        # Préparation des données
        X_input = scaler.transform([list(params.values())])
        y_pred = model.predict(X_input)[0][0]

        return JsonResponse({"throughput": round(y_pred, 2)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

def get_antennas(request):
    data = pd.read_csv("data/mm-5G-enriched.csv")

    # On filtre les points avec antenne NR active
    antennas = data[data["nrStatus"] == 1]

    # On garde la dernière valeur par tower_id
    antennas = antennas.groupby("tower_id").agg({
        'latitude': 'last',
        'longitude': 'last',
        'nr_ssRsrp': 'mean',
    }).reset_index()

    # Formule empirique : plus le RSRP est faible (ex: -110), plus le rayon est petit
    def estimate_radius(rsrp):
        return max(50, min(500, 200 + (rsrp + 90) * 5))  # entre 50m et 500m

    antennas["radius"] = antennas["nr_ssRsrp"].apply(estimate_radius)

    results = antennas.to_dict(orient="records")
    return JsonResponse(results, safe=False)
# API de prédiction
@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            values = [data.get(f, 0) for f in feature_names]
            X = scaler.transform([values])
            prediction = model.predict(X)[0][0]
            return JsonResponse({'prediction': float(prediction)})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'message': 'POST method required'}, status=400)

# API de détection de drift
@csrf_exempt
def detect_drift(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            new_features = sorted(list(data.keys()))
            original_features = sorted(feature_names)
            drift = set(original_features).symmetric_difference(set(new_features))

            if drift:
                return JsonResponse({'drift_detected': True, 'difference': list(drift)})
            else:
                return JsonResponse({'drift_detected': False})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'message': 'POST method required'}, status=400)

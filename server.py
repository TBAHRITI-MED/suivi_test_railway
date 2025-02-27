#!/usr/bin/env python3
"""
Description de l'outil :
-------------------------
Cet outil sert à déterminer des regroupements (clusters) de données géolocalisées
et à analyser les ralentissements (via OpenAI). Il propose aussi différentes
méthodes de voisinage (k plus proches voisins, voisinage itératif, etc.).

Il inclut :
- Réception et stockage des données (latitude, longitude, speed)
- Détection d'un ralentissement (vitesse < 80% de la moyenne)
- Analyse asynchrone via OpenAI (GPT-4)
- Clustering (k-means, DBSCAN, hiérarchique) + un clustering spatio-temporel
- Routes Flask pour récupérer les données, calculer un corridor, etc.

Notes :
- Variables d'environnement requises :
  - DATABASE_URL : URL PostgreSQL
  - OPENAI_API_KEY : Clé OpenAI
"""

import math
import csv
import json
import os
import time
import threading
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# Pour OpenAI
import openai
# Pour le clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin

# ---------------------------------------------------
# 1. Configuration de la base de données PostgreSQL
# ---------------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------------------------------------------
# Variables pour l'analyse OpenAI asynchrone
# ---------------------------------------------------
last_analysis_time = 0
ANALYSIS_INTERVAL = 60  # Analyser une fois par minute
current_explanation = "Pas d'analyse disponible."
analysis_in_progress = False

# ---------------------------------------------------
# 2. Modèle pour stocker les données des capteurs
# ---------------------------------------------------
class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    speed = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.Float, default=time.time)  # Timestamp

# Création de la base de données si elle n'existe pas encore
with app.app_context():
    db.create_all()

# ---------------------------------------------------
# 3. OpenAI : Configuration et analyse asynchrone
# ---------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ La variable d'environnement 'OPENAI_API_KEY' n'est pas définie !")

def analyser_ralentissement_async(speed, avg_speed):
    """
    Fonction asynchrone qui analyse le ralentissement via OpenAI
    sans bloquer l'application Flask.
    """
    global current_explanation, analysis_in_progress
    
    try:
        print(f"🚀 Analyse asynchrone lancée (speed={speed}, avg={avg_speed})")
        prompt = (
            f"La vitesse actuelle est {speed} m/s, alors que la moyenne est {avg_speed:.2f} m/s. "
            f"Pourquoi pourrait-il y avoir un ralentissement à cet endroit ? (Réponse courte)"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Tu es un expert en analyse de trafic, réponds brièvement."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        if "choices" in response and response["choices"]:
            result = response["choices"][0]["message"]["content"]
            current_explanation = result
            print(f"✅ Réponse OpenAI : {result}")
        else:
            current_explanation = "Aucune explication trouvée."
    except Exception as e:
        current_explanation = "Erreur d'analyse du ralentissement."
        print(f"❌ Erreur OpenAI : {e}")
    finally:
        analysis_in_progress = False

def check_and_analyze_slowdown(speed, avg_speed):
    """
    Vérifie s'il faut lancer une analyse (toutes les ANALYSIS_INTERVAL secondes)
    et la lance dans un thread séparé pour ne pas bloquer.
    """
    global last_analysis_time, analysis_in_progress
    current_time = time.time()

    if not analysis_in_progress and (current_time - last_analysis_time) > ANALYSIS_INTERVAL:
        analysis_in_progress = True
        last_analysis_time = current_time
        thr = threading.Thread(target=analyser_ralentissement_async, args=(speed, avg_speed))
        thr.daemon = True
        thr.start()
        return True
    return False

# ---------------------------------------------------
# 4. Route : Réception des données en temps réel
# ---------------------------------------------------
@app.route("/api/push_data", methods=["POST"])
def push_data():
    if not request.json:
        return jsonify({"error": "No JSON body"}), 400

    body = request.json
    # Si les données sont envoyées sous la clé "data", décoder la chaîne JSON imbriquée
    if "data" in body:
        try:
            body = json.loads(body["data"])
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        lat = float(body.get("latitude", 0))
        lon = float(body.get("longitude", 0))
        speed = float(body.get("speed", 0))
    except ValueError:
        return jsonify({"error": "Invalid numeric values"}), 400

    # Calcul de la vitesse moyenne globale
    total_speed = db.session.query(db.func.sum(SensorData.speed)).scalar() or 0
    total_count = db.session.query(SensorData).count() or 1
    avg_speed = total_speed / total_count

    # Détection d'un ralentissement : si la vitesse < 80% de la moyenne (et moyenne > 0)
    ralentissement = False
    if avg_speed > 0 and speed < avg_speed * 0.8:
        ralentissement = True
        print(f"⚠️ Ralentissement détecté ! Vitesse: {speed} m/s, Moy: {avg_speed:.2f} m/s")
        check_and_analyze_slowdown(speed, avg_speed)

    # Sauvegarde en base
    current_time = time.time()
    new_data = SensorData(latitude=lat, longitude=lon, speed=speed, timestamp=current_time)
    db.session.add(new_data)
    db.session.commit()
    print(f"📡 Nouveau point en BD: lat={lat}, lon={lon}, speed={speed}")

    resp = {
        "status": "Data saved",
        "current_speed": speed,
        "average_speed": avg_speed,
        "slowdown_detected": ralentissement
    }
    # Ajouter explication si besoin
    if ralentissement:
        resp["slowdown_explanation"] = current_explanation

    return jsonify(resp), 200

# ---------------------------------------------------
# 5. Route : Obtenir la dernière analyse
# ---------------------------------------------------
@app.route("/get_latest_analysis", methods=["GET"])
def get_latest_analysis():
    return jsonify({
        "explanation": current_explanation,
        "analysis_in_progress": analysis_in_progress
    })

# ---------------------------------------------------
# 6. Clustering (k-means, DBSCAN, agglomératif) sur lat/lon
# ---------------------------------------------------
@app.route("/cluster", methods=["POST"])
def cluster_data():
    """
    Effectuer un clustering sur les données latitude/longitude.
    Ex JSON d'entrée :
    {
      "algo": "kmeans",  // ou "dbscan", "agglomerative"
      "n_clusters": 3,
      "eps": 0.1,
      "min_samples": 5
    }
    """
    data = request.json or {}
    algo = data.get("algo", "kmeans").lower()

    # Récupérer toutes les données
    points_db = SensorData.query.all()
    coords = np.array([[p.latitude, p.longitude]
                       for p in points_db if p.latitude != 0 or p.longitude != 0])

    if coords.shape[0] < 2:
        return jsonify({"error": "Not enough valid data to cluster."}), 400

    try:
        if algo == "kmeans":
            n_clusters = data.get("n_clusters", 3)
            model = KMeans(n_clusters=n_clusters, n_init=10)
            labels = model.fit_predict(coords)
            centers = model.cluster_centers_.tolist()
            result_algo = "kmeans"
        elif algo == "dbscan":
            eps = data.get("eps", 0.1)
            min_samples = data.get("min_samples", 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(coords)
            centers = []
            result_algo = "dbscan"
        elif algo in ["agglomerative", "hierarchical"]:
            n_clusters = data.get("n_clusters", 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(coords)
            centers = []
            result_algo = "agglomerative"
        else:
            return jsonify({"error": f"Unsupported clustering algo: {algo}"}), 400

        results = []
        for i, (lat, lon) in enumerate(coords):
            results.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "label": int(labels[i])
            })

        return jsonify({
            "algo": result_algo,
            "clusters": results,
            "centers": centers
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------
# 7. Route pour un clustering spatio-temporel (latitude, longitude, alpha * time)
# ---------------------------------------------------
@app.route("/spatio_temporal_cluster", methods=["POST"])
def spatio_temporal_cluster():
    """
    Clustering DBSCAN sur (latitude, longitude, alpha * timestamp).
    Ex JSON d'entrée :
    {
      "algo": "dbscan",       // éventuellement
      "eps": 0.2,
      "min_samples": 5,
      "alpha": 0.0001
    }
    """
    data = request.json or {}
    eps = data.get("eps", 0.1)
    min_samples = data.get("min_samples", 5)
    alpha = data.get("alpha", 0.0001)

    # Récupération des points depuis la BD
    points_db = SensorData.query.all()
    coords_t = []
    for p in points_db:
        lat, lon, t = p.latitude, p.longitude, p.timestamp
        if lat == 0 and lon == 0:
            continue
        coords_t.append([lat, lon, alpha * t])  # dimension temps * alpha

    if len(coords_t) < 2:
        return jsonify({"error": "Not enough valid data."}), 400

    coords_t = np.array(coords_t)

    try:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(coords_t)

        results = []
        for i, label in enumerate(labels):
            lat, lon, scaled_time = coords_t[i]
            # reconvertir le temps
            original_time = scaled_time / alpha
            results.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "timestamp": float(original_time),
                "cluster": int(label)
            })

        return jsonify({
            "algo": "dbscan_spatio_temporal",
            "eps": eps,
            "min_samples": min_samples,
            "alpha": alpha,
            "clusters": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------
# 8. Routage du corridor (segment)
# ---------------------------------------------------
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 111320.0
    x = (lon - lon0) * R * math.cos(math.radians(lat0))
    y = (lat - lat0) * R
    return (x, y)

def distance_point_to_segment(px, py, ax, ay, bx, by):
    ABx = bx - ax
    ABy = by - ay
    APx = px - ax
    APy = py - ay
    AB2 = ABx*ABx + ABy*ABy
    if AB2 == 0:
        return math.hypot(px - ax, py - ay)
    t = (APx*ABx + APy*ABy) / AB2
    if t < 0:
        return math.hypot(px - ax, py - ay)
    elif t > 1:
        return math.hypot(px - bx, py - by)
    else:
        projx = ax + t*ABx
        projy = ay + t*ABy
        return math.hypot(px - projx, py - projy)

def is_point_on_segment(latP, lonP, latA, lonA, latB, lonB, corridor_width=30):
    lat0, lon0 = latA, lonA
    px, py = latlon_to_xy(latP, lonP, lat0, lon0)
    ax, ay = latlon_to_xy(latA, lonA, lat0, lon0)
    bx, by = latlon_to_xy(latB, lonB, lat0, lon0)
    dist = distance_point_to_segment(px, py, ax, ay, bx, by)
    return dist <= corridor_width

@app.route("/compute", methods=["POST"])
def compute():
    data = request.json
    latA = float(data["latA"])
    lonA = float(data["lonA"])
    latB = float(data["latB"])
    lonB = float(data["lonB"])
    corridor = 30.0

    onStreet = []
    offStreet = []
    speedSum = 0.0
    onCount = 0

    points = SensorData.query.all()
    for p in points:
        if is_point_on_segment(p.latitude, p.longitude, latA, lonA, latB, lonB, corridor):
            onStreet.append([p.latitude, p.longitude])
            speedSum += p.speed
            onCount += 1
        else:
            offStreet.append([p.latitude, p.longitude])
    avgSpeed = speedSum / onCount if onCount > 0 else 0.0
    return jsonify({
        "onStreet": onStreet,
        "offStreet": offStreet,
        "avgSpeed": avgSpeed
    })

@app.route("/compute_multiple", methods=["POST"])
def compute_multiple():
    data = request.json
    segments = data["segments"]
    results = []
    for idx, segment in enumerate(segments):
        latA, lonA = segment[0]
        latB, lonB = segment[1]
        onCount, offCount = compute_segment_points(latA, lonA, latB, lonB, 30.0)
        results.append({
            "segmentIndex": idx,
            "onCount": onCount,
            "offCount": offCount
        })
    if len(segments) > 1:
        latA, lonA = segments[0][0]
        latZ, lonZ = segments[-1][1]
        onCount, offCount = compute_segment_points(latA, lonA, latZ, lonZ, 30.0)
        results.append({
            "segmentIndex": "A->Z",
            "onCount": onCount,
            "offCount": offCount
        })
    return jsonify({"results": results})

def compute_segment_points(latA, lonA, latB, lonB, corridor=30.0):
    onCount = 0
    offCount = 0
    points = SensorData.query.all()
    for p in points:
        if is_point_on_segment(p.latitude, p.longitude, latA, lonA, latB, lonB, corridor):
            onCount += 1
        else:
            offCount += 1
    return onCount, offCount

# ---------------------------------------------------
# 9. Lancement du serveur Flask
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Render attribue un port dynamique
    app.run(host="0.0.0.0", port=port, debug=False)

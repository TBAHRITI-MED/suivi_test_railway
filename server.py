#!/usr/bin/env python3
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
from openai import OpenAI
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin

# ---------------------------------------------------
# 1. Configuration de la base de données PostgreSQL
# ---------------------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Variables globales pour l'analyse OpenAI asynchrone
last_analysis_time = 0
ANALYSIS_INTERVAL = 60  # Analyser une fois par minute
current_explanation = "Pas d'analyse disponible."
analysis_in_progress = False

# Variables globales pour ML
ml_models = {
    'isolation_forest': None,
    'dbscan': None,
    'is_trained': False
}

# ---------------------------------------------------
# 2. Modèle pour stocker les données des capteurs
# ---------------------------------------------------
class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    speed = db.Column(db.Float, nullable=False)

# Création de la base de données si elle n'existe pas encore
with app.app_context():
    db.create_all()

# ---------------------------------------------------
# 3. Fonctions d'analyse ML
# ---------------------------------------------------
def train_ml_models():
    """Entraîne les modèles ML avec les données existantes"""
    global ml_models
    
    try:
        # Récupérer toutes les données
        points = SensorData.query.all()
        
        if len(points) < 10:
            print("⚠️ Pas assez de données pour l'entraînement")
            return False
            
        # Préparer les données pour le clustering géographique
        coords = np.array([[p.latitude, p.longitude] for p in points])
        speeds = np.array([[p.speed] for p in points])
        
        # DBSCAN pour le clustering des zones
        eps_meters = 50  # 50 mètres entre points pour former un cluster
        kms_per_radian = 6371.0088  # Rayon de la Terre en km
        epsilon = eps_meters / 1000 / kms_per_radian
        
        # Entraîner DBSCAN
        dbscan = DBSCAN(
            eps=epsilon, 
            min_samples=5, 
            algorithm='ball_tree',
            metric='haversine'
        )
        dbscan.fit(np.radians(coords))
        
        # Isolation Forest pour détecter les anomalies de vitesse
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(speeds)
        
        # Sauvegarder les modèles en mémoire
        ml_models['isolation_forest'] = iso_forest
        ml_models['dbscan'] = dbscan
        ml_models['is_trained'] = True
        
        # Sauvegarder sur disque (optionnel)
        model_dir = "ml_models"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(iso_forest, os.path.join(model_dir, "iso_forest.pkl"))
        joblib.dump(dbscan, os.path.join(model_dir, "dbscan.pkl"))
        
        print(f"✅ Modèles ML entraînés avec succès sur {len(points)} points")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement ML: {e}")
        return False

def load_ml_models():
    """Tente de charger les modèles préexistants"""
    global ml_models
    
    try:
        model_dir = "ml_models"
        iso_path = os.path.join(model_dir, "iso_forest.pkl")
        dbscan_path = os.path.join(model_dir, "dbscan.pkl")
        
        if os.path.exists(iso_path) and os.path.exists(dbscan_path):
            ml_models['isolation_forest'] = joblib.load(iso_path)
            ml_models['dbscan'] = joblib.load(dbscan_path)
            ml_models['is_trained'] = True
            print("✅ Modèles ML chargés depuis les fichiers")
            return True
    except Exception as e:
        print(f"⚠️ Impossible de charger les modèles ML: {e}")
    
    return False

def analyze_point(lat, lon, speed):
    """Analyse un point avec les modèles ML"""
    if not ml_models['is_trained']:
        if not load_ml_models():
            # Tenter un entraînement si pas de modèles préexistants
            train_ml_models()
        
        if not ml_models['is_trained']:
            return {
                "status": "no_model",
                "message": "Aucun modèle ML disponible"
            }
    
    result = {"status": "success"}
    
    # Analyser la vitesse avec Isolation Forest
    if ml_models['isolation_forest'] is not None:
        # Prédiction d'anomalie
        speed_array = np.array([[speed]])
        is_anomaly = ml_models['isolation_forest'].predict(speed_array)[0] == -1
        anomaly_score = ml_models['isolation_forest'].score_samples(speed_array)[0]
        
        result["speed_analysis"] = {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "message": "Vitesse anormale détectée" if is_anomaly else "Vitesse normale"
        }
    
    # Trouver les zones populaires
    points = SensorData.query.all()
    if len(points) >= 10:
        clusters = {}
        coords = np.array([[p.latitude, p.longitude] for p in points])
        
        if ml_models['dbscan'] is not None:
            # Appliquer les labels de cluster à tous les points
            labels = ml_models['dbscan'].labels_
            
            # Compter les points par cluster et trouver les centres
            for i, label in enumerate(labels):
                if label != -1:  # Ignorer les points considérés comme du bruit
                    if label not in clusters:
                        clusters[label] = {
                            "count": 0,
                            "lat_sum": 0,
                            "lon_sum": 0,
                            "points": []
                        }
                    clusters[label]["count"] += 1
                    clusters[label]["lat_sum"] += coords[i][0]
                    clusters[label]["lon_sum"] += coords[i][1]
                    clusters[label]["points"].append(coords[i])
            
            # Calculer les centres de cluster
            popular_zones = []
            for label, data in clusters.items():
                if data["count"] >= 5:  # Au moins 5 points
                    center_lat = data["lat_sum"] / data["count"]
                    center_lon = data["lon_sum"] / data["count"]
                    popular_zones.append({
                        "cluster_id": int(label),
                        "center": [float(center_lat), float(center_lon)],
                        "point_count": data["count"]
                    })
            
            # Trier par nombre de points
            popular_zones.sort(key=lambda x: x["point_count"], reverse=True)
            
            # Garder les 3 premiers
            result["popular_zones"] = popular_zones[:3]
            
            # Vérifier si le point actuel est dans une zone populaire
            is_in_zone = False
            for zone in popular_zones[:3]:
                # Distance approximative
                dist = haversine_distance(lat, lon, zone["center"][0], zone["center"][1])
                if dist < 50:  # 50 mètres
                    is_in_zone = True
                    result["current_zone"] = {
                        "in_popular_zone": True,
                        "zone_rank": popular_zones.index(zone) + 1,
                        "cluster_id": zone["cluster_id"],
                        "point_count": zone["point_count"]
                    }
                    break
            
            if not is_in_zone:
                result["current_zone"] = {"in_popular_zone": False}
    
    return result

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points en mètres"""
    # Convertir en radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Formule de haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Rayon de la Terre en km
    return km * 1000  # En mètres

def analyser_ralentissement_async(speed, avg_speed):
    """
    Fonction asynchrone qui analyse le ralentissement sans bloquer l'application
    """
    global current_explanation, analysis_in_progress
    
    try:
        print(f"🚀 Analyse asynchrone lancée avec speed={speed}, avg_speed={avg_speed}")
        
        # Initialiser le client OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Appel API avec un timeout court et un prompt simplifié
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "Tu es un expert trafic. Réponse courte (30 mots max)."},
                {"role": "user", "content": f"Pourquoi vitesse={speed}m/s vs moyenne={avg_speed:.2f}m/s? Raison courte."}
            ],
            max_tokens=50
        )
        
        # Extraction de la réponse
        if hasattr(response, 'choices') and response.choices:
            result = response.choices[0].message.content
            current_explanation = result
            print(f"✅ Réponse OpenAI: {result}")
        else:
            current_explanation = "Aucune explication trouvée."
            
    except Exception as e:
        current_explanation = "Erreur d'analyse."
        print(f"❌ Erreur OpenAI: {e}")
    finally:
        analysis_in_progress = False

def check_and_analyze_slowdown(speed, avg_speed):
    """
    Vérifie s'il faut lancer une analyse et la lance dans un thread séparé
    """
    global last_analysis_time, analysis_in_progress
    
    current_time = time.time()
    
    # Lancer l'analyse seulement si le temps minimum est passé
    if not analysis_in_progress and (current_time - last_analysis_time) > ANALYSIS_INTERVAL:
        analysis_in_progress = True
        last_analysis_time = current_time
        
        # Lancer l'analyse dans un thread séparé
        thread = threading.Thread(
            target=analyser_ralentissement_async, 
            args=(speed, avg_speed)
        )
        thread.daemon = True  # Le thread se terminera quand le programme principal se termine
        thread.start()
        
        return True
    return False

# ---------------------------------------------------
# 4. Route pour recevoir et stocker les données en temps réel
# ---------------------------------------------------
@app.route("/api/push_data", methods=["POST"])
def push_data():
    if not request.json:
        return jsonify({"error": "No JSON body"}), 400

    body = request.json

    # Si les données sont envoyées sous la clé "data", décoder la chaîne JSON imbriquée
    if "data" in body:
        try:
            if isinstance(body["data"], str):
                body = json.loads(body["data"])
            else:
                body = body["data"]
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        lat = float(body.get("latitude", 0))
        lon = float(body.get("longitude", 0))
        speed = float(body.get("speed", 0))
    except ValueError:
        return jsonify({"error": "Invalid numeric values"}), 400

    # Calcul de la vitesse moyenne actuelle (pour info)
    total_speed = db.session.query(db.func.sum(SensorData.speed)).scalar() or 0
    total_count = db.session.query(SensorData).count() or 1  # éviter division par zéro
    avg_speed = total_speed / total_count

    # Détection d'un ralentissement (si speed < 80% de la moyenne et que la moyenne > 0)
    ralentissement = False
    if avg_speed > 0 and speed < avg_speed * 0.8:
        ralentissement = True
        print(f"⚠️ Ralentissement détecté ! Vitesse actuelle: {speed} m/s, Moyenne: {avg_speed:.2f} m/s")
        
        # Lancer l'analyse en arrière-plan (ne bloque pas)
        check_and_analyze_slowdown(speed, avg_speed)

    # Sauvegarde en base de données
    new_data = SensorData(latitude=lat, longitude=lon, speed=speed)
    db.session.add(new_data)
    db.session.commit()
    print(f"📡 Nouveau point ajouté en BD: lat={lat}, lon={lon}, speed={speed}")

    response_data = {
        "status": "Data saved",
        "current_speed": speed,
        "average_speed": avg_speed,
        "slowdown_detected": ralentissement
    }
    
    # Ajouter l'explication si disponible et s'il y a ralentissement
    if ralentissement:
        response_data["slowdown_explanation"] = current_explanation
    
    # Analyse ML du point
    ml_results = analyze_point(lat, lon, speed)
    if ml_results["status"] == "success":
        response_data["ml_analysis"] = ml_results
    
    return jsonify(response_data), 200

# ---------------------------------------------------
# 5. Routes pour la gestion des modèles ML
# ---------------------------------------------------
@app.route("/ml/train", methods=["POST"])
def train_models_route():
    """Route pour entraîner ou réentraîner les modèles ML"""
    success = train_ml_models()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Modèles ML entraînés avec succès"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Erreur lors de l'entraînement des modèles"
        }), 500

@app.route("/ml/status", methods=["GET"])
def get_ml_status():
    """Route pour obtenir l'état des modèles ML"""
    return jsonify({
        "models_trained": ml_models["is_trained"],
        "isolation_forest_available": ml_models["isolation_forest"] is not None,
        "dbscan_available": ml_models["dbscan"] is not None
    })

@app.route("/ml/export_data", methods=["GET"])
def export_data():
    """Route pour exporter les données en CSV"""
    try:
        # Récupérer tous les points
        points = SensorData.query.all()
        
        if len(points) == 0:
            return jsonify({"error": "No data to export"}), 404
            
        # Créer un DataFrame pandas
        data = []
        for p in points:
            data.append({
                "id": p.id,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "speed": p.speed
            })
            
        df = pd.DataFrame(data)
        
        # Créer le répertoire d'export
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Créer un nom de fichier unique
        filename = f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(export_dir, filename)
        
        # Exporter en CSV
        df.to_csv(filepath, index=False)
        
        return jsonify({
            "status": "success",
            "message": f"{len(points)} points exportés",
            "filename": filename,
            "path": filepath
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/ml/analyze_point", methods=["POST"])
def analyze_point_route():
    """Route pour analyser un point spécifique"""
    if not request.json:
        return jsonify({"error": "No JSON body"}), 400
        
    try:
        body = request.json
        lat = float(body.get("latitude", 0))
        lon = float(body.get("longitude", 0))
        speed = float(body.get("speed", 0))
        
        result = analyze_point(lat, lon, speed)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/ml/popular_zones", methods=["GET"])
def get_popular_zones():
    """Route pour obtenir les zones populaires"""
    try:
        # Récupérer tous les points
        points = SensorData.query.all()
        
        if len(points) < 10:
            return jsonify({
                "status": "error",
                "message": f"Pas assez de données ({len(points)} points)"
            }), 400
            
        # Calculer les clusters si le modèle est disponible
        if ml_models["is_trained"] and ml_models["dbscan"] is not None:
            # Préparer les coordonnées
            coords = np.array([[p.latitude, p.longitude] for p in points])
            
            # Appliquer les labels de cluster
            labels = ml_models["dbscan"].labels_
            
            # Compter les points par cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label != -1:  # Ignorer le bruit
                    if label not in clusters:
                        clusters[label] = {
                            "count": 0,
                            "lat_sum": 0,
                            "lon_sum": 0
                        }
                    clusters[label]["count"] += 1
                    clusters[label]["lat_sum"] += coords[i][0]
                    clusters[label]["lon_sum"] += coords[i][1]
            
            # Calculer les centres et préparer le résultat
            result = []
            for label, data in clusters.items():
                center_lat = data["lat_sum"] / data["count"]
                center_lon = data["lon_sum"] / data["count"]
                result.append({
                    "cluster_id": int(label),
                    "center": [float(center_lat), float(center_lon)],
                    "point_count": data["count"]
                })
            
            # Trier par nombre de points
            result.sort(key=lambda x: x["point_count"], reverse=True)
            
            return jsonify({
                "status": "success",
                "zones": result,
                "total_clusters": len(result)
            })
            
        else:
            # Fallback: méthode simple de grille
            min_lat = min(p.latitude for p in points)
            max_lat = max(p.latitude for p in points)
            min_lon = min(p.longitude for p in points)
            max_lon = max(p.longitude for p in points)
            
            # Grille 5x5
            lat_step = (max_lat - min_lat) / 5
            lon_step = (max_lon - min_lon) / 5
            
            grid = {}
            for p in points:
                i = min(4, max(0, int((p.latitude - min_lat) / lat_step)))
                j = min(4, max(0, int((p.longitude - min_lon) / lon_step)))
                key = f"{i}_{j}"
                
                if key not in grid:
                    grid[key] = {
                        "count": 0,
                        "lat_sum": 0,
                        "lon_sum": 0
                    }
                
                grid[key]["count"] += 1
                grid[key]["lat_sum"] += p.latitude
                grid[key]["lon_sum"] += p.longitude
            
            # Convertir en liste
            zones = []
            for key, data in grid.items():
                if data["count"] >= 5:  # Au moins 5 points
                    i, j = map(int, key.split('_'))
                    center_lat = data["lat_sum"] / data["count"]
                    center_lon = data["lon_sum"] / data["count"]
                    
                    zones.append({
                        "grid_cell": key,
                        "center": [float(center_lat), float(center_lon)],
                        "point_count": data["count"]
                    })
            
            # Trier par nombre de points
            zones.sort(key=lambda x: x["point_count"], reverse=True)
            
            return jsonify({
                "status": "success",
                "zones": zones[:5],  # Top 5
                "method": "grid"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/ml/speed_anomalies", methods=["GET"])
def get_speed_anomalies():
    """Route pour obtenir les anomalies de vitesse"""
    try:
        # Vérifier si le modèle est entraîné
        if not ml_models["is_trained"] or ml_models["isolation_forest"] is None:
            if not train_ml_models():
                return jsonify({
                    "status": "error",
                    "message": "Modèle d'anomalies non disponible"
                }), 400
        
        # Récupérer tous les points
        points = SensorData.query.all()
        
        if len(points) < 10:
            return jsonify({
                "status": "error",
                "message": f"Pas assez de données ({len(points)} points)"
            }), 400
        
        # Analyser tous les points
        speeds = np.array([[p.speed] for p in points])
        predictions = ml_models["isolation_forest"].predict(speeds)
        scores = ml_models["isolation_forest"].score_samples(speeds)
        
        # Collecter les anomalies
        anomalies = []
        for i, (p, pred, score) in enumerate(zip(points, predictions, scores)):
            if pred == -1:  # C'est une anomalie
                anomalies.append({
                    "id": p.id,
                    "latitude": float(p.latitude),
                    "longitude": float(p.longitude),
                    "speed": float(p.speed),
                    "anomaly_score": float(score)
                })
        
        # Trier par score d'anomalie (plus négatif = plus anormal)
        anomalies.sort(key=lambda x: x["anomaly_score"])
        
        return jsonify({
            "status": "success",
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "total_points": len(points)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ---------------------------------------------------
# 6. Route pour obtenir la dernière analyse
# ---------------------------------------------------
@app.route("/get_latest_analysis", methods=["GET"])
def get_latest_analysis():
    return jsonify({
        "explanation": current_explanation,
        "analysis_in_progress": analysis_in_progress
    })

# ---------------------------------------------------
# 7. Fonctions utilitaires : distance point-segment
# ---------------------------------------------------
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 111320.0  # Nombre de mètres par degré approximativement
    x = (lon - lon0) * R * math.cos(math.radians(lat0))
    y = (lat - lat0) * R
    return (x, y)

def distance_point_to_segment(px, py, ax, ay, bx, by):
    ABx = bx - ax
    ABy = by - ay
    APx = px - ax
    APy = py - ay
    AB2 = ABx * ABx + ABy * ABy
    if AB2 == 0:
        return math.hypot(px - ax, py - ay)
    t = (APx * ABx + APy * ABy) / AB2
    if t < 0:
        return math.hypot(px - ax, py - ay)
    elif t > 1:
        return math.hypot(px - bx, py - by)
    else:
        projx = ax + t * ABx
        projy = ay + t * ABy
        return math.hypot(px - projx, py - projy)

def is_point_on_segment(latP, lonP, latA, lonA, latB, lonB, corridor_width=30):
    lat0, lon0 = latA, lonA
    px, py = latlon_to_xy(latP, lonP, lat0, lon0)
    ax, ay = latlon_to_xy(latA, lonA, lat0, lon0)
    bx, by = latlon_to_xy(latB, lonB, lat0, lon0)
    dist = distance_point_to_segment(px, py, ax, ay, bx, by)
    return dist <= corridor_width

# ---------------------------------------------------
# 8. Routes pour les requêtes et analyses
# ---------------------------------------------------
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/get_all_points", methods=["GET"])
def get_all_points():
    points = SensorData.query.all()
    data = [{"latitude": p.latitude, "longitude": p.longitude, "speed": p.speed} for p in points]
    return jsonify({"points": data})

@app.route("/get_latest_point", methods=["GET"])
def get_latest_point():
    """Récupère le dernier point enregistré"""
    try:
        latest = SensorData.query.order_by(SensorData.id.desc()).first()
        if latest:
            # Obtenir les résultats ML
            ml_results = analyze_point(latest.latitude, latest.longitude, latest.speed)
            
            return jsonify({
                "status": "success",
                "latest_point": {
                    "latitude": latest.latitude,
                    "longitude": latest.longitude,
                    "speed": latest.speed,
                    "is_latest": True
                },
                "ml_analysis": ml_results
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Aucun point trouvé"
            }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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
# 9. Initialisation au démarrage
# ---------------------------------------------------
def initialize_app():
    """Initialise l'application au démarrage"""
    print("🚀 Initialisation de l'application...")
    
    # Charger les modèles préexistants
    if not load_ml_models():
        print("⚠️ Aucun modèle ML préexistant trouvé")
        
        # Vérifier s'il y a assez de données pour entraîner
        with app.app_context():
            count = SensorData.query.count()
            if count >= 10:
                print(f"📊 {count} points trouvés en base de données, tentative d'entraînement automatique...")
                if train_ml_models():
                    print("✅ Modèles ML entraînés automatiquement au démarrage")
                else:
                    print("❌ Échec de l'entraînement automatique")
            else:
                print(f"⚠️ Pas assez de données pour l'entraînement ({count} points trouvés)")
    else:
        print("✅ Modèles ML chargés avec succès")

# ---------------------------------------------------
# 10. Lancement du serveur Flask sur Render
# ---------------------------------------------------
if __name__ == "__main__":
    # Initialiser l'application
    initialize_app()
    
    # Démarrer le serveur
    port = int(os.environ.get("PORT", 5001))  # Render attribue un port dynamique
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # Si importé comme module (par Gunicorn par exemple)
    initialize_app()
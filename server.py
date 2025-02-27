#!/usr/bin/env python3
import math
import csv
import json
import os
import time
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI

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
# 3. Fonctions d'analyse OpenAI asynchrones
# ---------------------------------------------------
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
            body = json.loads(body["data"])
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
    
    return jsonify(response_data), 200

# ---------------------------------------------------
# 5. Route pour obtenir la dernière analyse
# ---------------------------------------------------
@app.route("/get_latest_analysis", methods=["GET"])
def get_latest_analysis():
    return jsonify({
        "explanation": current_explanation,
        "analysis_in_progress": analysis_in_progress
    })

# ---------------------------------------------------
# 6. Fonctions utilitaires : distance point-segment
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
# 7. Routes pour les requêtes et analyses
# ---------------------------------------------------
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/get_all_points", methods=["GET"])
def get_all_points():
    points = SensorData.query.all()
    data = [{"latitude": p.latitude, "longitude": p.longitude, "speed": p.speed} for p in points]
    return jsonify({"points": data})

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
# 8. Lancement du serveur Flask sur Render
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Render attribue un port dynamique
    app.run(host="0.0.0.0", port=port, debug=False)
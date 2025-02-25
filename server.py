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
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration de la base de données
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Variables globales pour l'analyse
last_analysis_time = 0
ANALYSIS_INTERVAL = 60  # Analyser seulement une fois par minute
current_explanation = "Analyse en attente..."
analysis_in_progress = False

# Modèle pour stocker les données des capteurs
class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    speed = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Création de la base de données
with app.app_context():
    db.create_all()

# Fonction d'analyse OpenAI exécutée dans un thread séparé
def analyser_ralentissement_async(speed, avg_speed):
    global current_explanation, analysis_in_progress
    
    try:
        print(f"🚀 Analyse asynchrone lancée avec speed={speed}, avg_speed={avg_speed}")
        
        # Prompt court pour obtenir une réponse rapide
        prompt = f"Pourquoi vitesse={speed}m/s vs moyenne={avg_speed:.2f}m/s? Réponse courte (max 30 mots)"
        
        # Initialiser le client OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Appel API avec un timeout court
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Modèle plus rapide que GPT-4
            messages=[
                {"role": "system", "content": "Expert trafic. Réponds simplement en 30 mots max."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60  # Limiter la longueur de la réponse
        )
        
        # Extraction de la réponse
        if hasattr(response, 'choices') and response.choices:
            result = response.choices[0].message.content
            current_explanation = result
        else:
            current_explanation = "Impossible d'analyser le ralentissement."
            
    except Exception as e:
        current_explanation = f"Erreur: {str(e)[:50]}..."
        print(f"❌ Erreur OpenAI: {e}")
    finally:
        analysis_in_progress = False

# Route pour recevoir et stocker les données
@app.route("/api/push_data", methods=["POST"])
def push_data():
    global last_analysis_time, analysis_in_progress, current_explanation
    
    if not request.json:
        return jsonify({"error": "No JSON body"}), 400

    body = request.json
    if "data" in body:
        try:
            body = json.loads(body["data"])
        except json.JSONDecodeError as e:
            return jsonify({"error": "Invalid JSON format"}), 400

    try:
        lat = float(body.get("latitude", 0))
        lon = float(body.get("longitude", 0))
        speed = float(body.get("speed", 0))
    except ValueError:
        return jsonify({"error": "Invalid numeric values"}), 400

    # Sauvegarde en base de données
    new_data = SensorData(latitude=lat, longitude=lon, speed=speed)
    db.session.add(new_data)
    db.session.commit()
    print(f"📡 Point ajouté: lat={lat}, lon={lon}, speed={speed}")

    # Calcul de vitesse moyenne
    # Utiliser seulement les points des dernières minutes pour plus de pertinence
    current_time = time.time()
    one_minute_ago = datetime.utcnow().timestamp() - 60
    
    recent_points = SensorData.query.filter(
        SensorData.timestamp >= datetime.fromtimestamp(one_minute_ago)
    ).all()
    
    speeds = [p.speed for p in recent_points if p.speed > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    
    # Détection ralentissement
    ralentissement = False
    if avg_speed > 0 and speed < avg_speed * 0.8:
        ralentissement = True
        print(f"⚠️ Ralentissement: {speed} m/s vs {avg_speed:.2f} m/s")
        
        # Lancer l'analyse seulement si:
        # 1. Aucune analyse n'est en cours
        # 2. La dernière analyse date de plus d'une minute
        if not analysis_in_progress and (current_time - last_analysis_time) > ANALYSIS_INTERVAL:
            analysis_in_progress = True
            last_analysis_time = current_time
            
            # Lancer l'analyse dans un thread séparé
            threading.Thread(
                target=analyser_ralentissement_async, 
                args=(speed, avg_speed)
            ).start()

    # Réponse simple et rapide
    response_data = {
        "status": "Data saved",
        "current_speed": speed,
        "average_speed": round(avg_speed, 2),
        "slowdown_detected": ralentissement
    }
    
    # Ajouter l'explication actuelle si ralentissement
    if ralentissement:
        response_data["slowdown_explanation"] = current_explanation
    
    return jsonify(response_data), 200

# Route pour obtenir l'explication actuelle
@app.route("/get_explanation", methods=["GET"])
def get_explanation():
    return jsonify({
        "explanation": current_explanation,
        "analysis_in_progress": analysis_in_progress
    })

# Routes existantes...
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/get_all_points", methods=["GET"])
def get_all_points():
    points = SensorData.query.all()
    data = [{"latitude": p.latitude, "longitude": p.longitude, "speed": p.speed} for p in points]
    return jsonify({"points": data})

# Lancement du serveur
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
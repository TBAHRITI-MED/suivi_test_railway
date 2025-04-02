import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flask import Blueprint

# Créer un Blueprint Flask pour les fonctionnalités ML
ml_routes = Blueprint('ml_routes', __name__)

class MovementAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, data_df):
        """
        Entraîne le modèle de détection d'anomalies
        
        Args:
            data_df: DataFrame contenant au moins les colonnes 'speed', 'accelerationX', 'accelerationY', 'accelerationZ'
        """
        # Sélectionner les caractéristiques pertinentes
        features = data_df[['speed', 'accelerationX', 'accelerationY', 'accelerationZ']]
        
        # Normaliser les données
        scaled_features = self.scaler.fit_transform(features)
        
        # Entraîner le modèle
        self.model.fit(scaled_features)
        self.is_trained = True
        print("✅ Modèle de détection d'anomalies entraîné avec succès")
        
    def predict(self, speed, accel_x, accel_y, accel_z):
        """
        Prédit si un point de données est une anomalie
        
        Returns:
            -1 si anomalie, 1 sinon
        """
        if not self.is_trained:
            print("⚠️ Le modèle n'est pas encore entraîné !")
            return 0
            
        # Créer le vecteur de caractéristiques et le normaliser
        features = np.array([[speed, accel_x, accel_y, accel_z]])
        scaled_features = self.scaler.transform(features)
        
        # Prédiction
        prediction = self.model.predict(scaled_features)[0]
        
        return prediction
        
    def get_anomaly_score(self, speed, accel_x, accel_y, accel_z):
        """
        Retourne un score d'anomalie (plus négatif = plus anormal)
        """
        if not self.is_trained:
            return 0
            
        features = np.array([[speed, accel_x, accel_y, accel_z]])
        scaled_features = self.scaler.transform(features)
        
        # Score d'anomalie
        score = self.model.score_samples(scaled_features)[0]
        
        return score

# Créer une instance du détecteur
anomaly_detector = MovementAnomalyDetector()

# Fonction pour analyser les mouvements et détecter une chute potentielle
def analyze_movement_patterns(speed, accel_x, accel_y, accel_z):
    """
    Analyse les mouvements pour détecter des comportements anormaux
    
    Args:
        speed: vitesse en m/s
        accel_x,y,z: accélération selon les 3 axes
        
    Returns:
        dict: Résultat de l'analyse avec type d'événement et confiance
    """
    # Calcul de magnitude d'accélération
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    # Règles simples pour la détection de chute (avant l'entraînement du modèle)
    if not anomaly_detector.is_trained:
        # Chute possible: forte décélération combinée à une accélération élevée
        if speed < 0.5 and accel_magnitude > 2.5:
            return {
                "event_type": "possible_fall",
                "confidence": min(0.6, accel_magnitude / 5.0),
                "details": "Forte décélération et accélération élevée détectées"
            }
        return {"event_type": "normal", "confidence": 0.9}
    
    # Utilisation du modèle entraîné
    prediction = anomaly_detector.predict(speed, accel_x, accel_y, accel_z)
    score = anomaly_detector.get_anomaly_score(speed, accel_x, accel_y, accel_z)
    
    if prediction == -1:
        # Anomalie détectée
        # Plus le score est négatif, plus l'anomalie est forte
        confidence = min(0.95, 0.5 - score)  # Convertir score en confiance entre 0.5 et 0.95
        
        # Classifier le type d'anomalie
        if accel_magnitude > 3.0:
            event_type = "possible_fall"
            details = "Accélération anormale (possible chute)"
        elif speed < 0.2 and accel_magnitude > 1.5:
            event_type = "sudden_stop"
            details = "Arrêt soudain"
        else:
            event_type = "unusual_movement"
            details = "Mouvement inhabituel"
            
        return {
            "event_type": event_type,
            "confidence": float(confidence),
            "details": details,
            "anomaly_score": float(score)
        }
    
    return {"event_type": "normal", "confidence": 0.9, "anomaly_score": float(score)}

# ---------------------------------------------------
# Routes Flask pour l'API de ML
# ---------------------------------------------------

from flask import request, jsonify

@ml_routes.route("/train_anomaly_model", methods=["POST"])
def train_anomaly_model_route():
    """Route pour entraîner le modèle avec les données existantes"""
    from app import db, SensorData  # Import local pour éviter les imports circulaires
    
    try:
        # Récupérer toutes les données
        data = SensorData.query.all()
        
        # Vérifier si on a assez de données
        if len(data) < 50:
            return jsonify({
                "status": "error",
                "message": f"Pas assez de données pour l'entraînement ({len(data)} points). Minimum 50 requis."
            }), 400
        
        # Créer un DataFrame
        data_list = []
        for point in data:
            # Adapter selon votre schéma de données réel
            data_list.append({
                "speed": point.speed,
                "accelerationX": 0,  # À adapter selon votre schéma
                "accelerationY": 0,  # À adapter selon votre schéma
                "accelerationZ": 0   # À adapter selon votre schéma
            })
        
        df = pd.DataFrame(data_list)
        
        # Entraîner le modèle
        anomaly_detector.train(df)
        
        return jsonify({
            "status": "success",
            "message": f"Modèle entraîné avec succès sur {len(data)} points de données"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de l'entraînement: {str(e)}"
        }), 500

@ml_routes.route("/analyze_movement", methods=["POST"])
def analyze_movement_route():
    """Route pour analyser un mouvement spécifique"""
    if not request.json:
        return jsonify({"error": "No JSON body"}), 400
        
    try:
        body = request.json
        speed = float(body.get("speed", 0))
        accel_x = float(body.get("accelerationX", 0))
        accel_y = float(body.get("accelerationY", 0))
        accel_z = float(body.get("accelerationZ", 0))
        
        # Analyser le mouvement
        result = analyze_movement_patterns(speed, accel_x, accel_y, accel_z)
        
        return jsonify({
            "status": "success",
            "analysis": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Erreur d'analyse: {str(e)}"
        }), 500
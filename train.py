import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import joblib

# Charger les données
df = pd.read_csv('sensor_data.csv')

# === CLUSTERING GÉOGRAPHIQUE ===
# Préparation des données
coords = df[['latitude', 'longitude']].values

# Conversion des coordonnées en radians pour DBSCAN
coords_rad = np.radians(coords)

# Clustering avec DBSCAN
kms_per_radian = 6371.0088  # Rayon de la Terre en km
eps = 50 / 1000 / kms_per_radian  # 50 mètres en radians
db = DBSCAN(eps=eps, min_samples=5, algorithm='ball_tree', metric='haversine').fit(coords_rad)

# Sauvegarder les étiquettes
df['cluster'] = db.labels_

# === DÉTECTION D'ANOMALIES ===
# Préparation des données pour l'isolation forest
features = df[['speed']].fillna(0)

# Entraînement du modèle de détection d'anomalies
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(features)

# Sauvegarder les modèles
joblib.dump(db, 'dbscan_model.pkl')
joblib.dump(iso_forest, 'isoforest_model.pkl')

# Sauvegarder également les paramètres
import json
with open('model_params.json', 'w') as f:
    json.dump({
        'eps_meters': 50,
        'min_samples': 5,
        'kms_per_radian': kms_per_radian
    }, f)

print("Modèles entraînés et sauvegardés.")
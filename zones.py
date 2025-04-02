import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from flask import Blueprint, jsonify, request
import json

geo_analytics = Blueprint('geo_analytics', __name__)

class GeoClusterAnalyzer:
    def __init__(self):
        self.eps = 50  # Distance maximale en mètres entre deux points pour être dans le même cluster
        self.min_samples = 5  # Nombre minimal de points pour former un cluster
        self.clusters = None
        self.cluster_centers = None
        
    def cluster_points(self, df):
        """
        Groupe les points GPS en clusters
        
        Args:
            df: DataFrame avec colonnes 'latitude' et 'longitude'
            
        Returns:
            df avec une colonne 'cluster' ajoutée
        """
        # Convertir le dataframe en matrice de coordonnées
        coords = df[['latitude', 'longitude']].values
        
        # Initialiser DBSCAN
        kms_per_radian = 6371.0088  # Rayon de la Terre en km
        epsilon = self.eps / 1000 / kms_per_radian  # Conversion de mètres en radians
        
        # Clustering
        db = DBSCAN(eps=epsilon, min_samples=self.min_samples, algorithm='ball_tree', 
                   metric='haversine').fit(np.radians(coords))
        
        # Ajouter les labels de cluster au dataframe
        df_result = df.copy()
        df_result['cluster'] = db.labels_
        
        # Stocker les résultats
        self.clusters = df_result
        self.compute_cluster_centers()
        
        return df_result
    
    def compute_cluster_centers(self):
        """Calcule les centres de chaque cluster"""
        if self.clusters is None:
            return None
            
        centers = {}
        for cluster_id in self.clusters['cluster'].unique():
            if cluster_id == -1:  # Ignorer les points considérés comme bruit
                continue
                
            # Points du cluster actuel
            points = self.clusters[self.clusters['cluster'] == cluster_id]
            
            if len(points) == 0:
                continue
                
            # Convertir en tuples (lat, lon)
            point_tuples = [(row.latitude, row.longitude) for _, row in points.iterrows()]
            
            # Calculer le centre du cluster (point central)
            center_point = self.get_centermost_point(point_tuples)
            centers[int(cluster_id)] = {
                'center': center_point,
                'num_points': len(points),
                'avg_speed': points['speed'].mean() if 'speed' in points.columns else None
            }
            
        self.cluster_centers = centers
        return centers
            
    def get_centermost_point(self, cluster_points):
        """
        Trouve le point le plus central d'un cluster
        """
        if len(cluster_points) == 1:
            return cluster_points[0]
            
        # Calculer toutes les distances à tous les autres points
        centroid = (sum(p[0] for p in cluster_points) / len(cluster_points),
                   sum(p[1] for p in cluster_points) / len(cluster_points))
        
        # Trouver le point le plus proche du centroïde
        centermost_point = min(cluster_points, 
                              key=lambda point: great_circle(point, centroid).m)
        
        return centermost_point
    
    def get_heatmap_data(self):
        """
        Prépare les données pour une carte de chaleur
        
        Returns:
            Liste de points avec leur poids pour visualisation
        """
        if self.clusters is None:
            return []
            
        # Compter les occurrences de chaque combinaison latitude/longitude
        grouped = self.clusters.groupby(['latitude', 'longitude']).size().reset_index(name='weight')
        
        # Formater pour la carte de chaleur
        heatmap_data = []
        for _, row in grouped.iterrows():
            heatmap_data.append({
                'lat': float(row['latitude']),
                'lng': float(row['longitude']),
                'weight': int(row['weight'])
            })
            
        return heatmap_data
    
    def get_popular_areas(self, top_n=5):
        """
        Retourne les zones les plus fréquentées
        """
        if self.cluster_centers is None:
            return []
            
        # Trier les clusters par nombre de points
        sorted_clusters = sorted(
            self.cluster_centers.items(),
            key=lambda x: x[1]['num_points'],
            reverse=True
        )
        
        # Prendre les top_n premiers
        result = []
        for i, (cluster_id, data) in enumerate(sorted_clusters[:top_n]):
            center = data['center']
            result.append({
                'rank': i + 1,
                'location': {'lat': center[0], 'lng': center[1]},
                'visits': data['num_points'],
                'avg_speed': data['avg_speed']
            })
            
        return result

# Créer une instance de l'analyseur
geo_analyzer = GeoClusterAnalyzer()

# ---------------------------------------------------
# Routes Flask pour l'API d'analyse géographique
# ---------------------------------------------------

@geo_analytics.route("/analyze_locations", methods=["GET"])
def analyze_locations():
    """
    Analyse toutes les localisations en base de données
    et retourne les zones populaires et données de heatmap
    """
    from app import db, SensorData  # Import local pour éviter les imports circulaires
    
    try:
        # Récupérer tous les points de localisation
        points = SensorData.query.all()
        
        # Vérifier si on a suffisamment de données
        if len(points) < 10:
            return jsonify({
                "status": "error",
                "message": f"Pas assez de données pour l'analyse ({len(points)} points). Minimum 10 requis."
            }), 400
        
        # Créer un DataFrame avec les points
        data = [{"latitude": p.latitude, "longitude": p.longitude, "speed": p.speed} for p in points]
        df = pd.DataFrame(data)
        
        # Analyser les clusters
        geo_analyzer.cluster_points(df)
        
        # Obtenir les résultats
        popular_areas = geo_analyzer.get_popular_areas()
        heatmap_data = geo_analyzer.get_heatmap_data()
        
        return jsonify({
            "status": "success",
            "popular_areas": popular_areas,
            "heatmap_data": heatmap_data[:1000],  # Limiter pour des raisons de performance
            "total_points": len(points),
            "total_clusters": sum(1 for c in geo_analyzer.clusters['cluster'].unique() if c != -1)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de l'analyse: {str(e)}"
        }), 500


@geo_analytics.route("/heatmap_data", methods=["GET"])
def get_heatmap_data():
    """Retourne les données de la carte de chaleur"""
    try:
        heatmap_data = geo_analyzer.get_heatmap_data()
        return jsonify({
            "status": "success",
            "heatmap_data": heatmap_data[:2000]  # Limiter pour éviter les données trop volumineuses
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
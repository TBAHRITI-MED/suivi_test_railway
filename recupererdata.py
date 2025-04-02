import psycopg2
import pandas as pd
import csv

# Connexion à la base de données
conn_string = "postgresql://data_suivi_user:oYFlVBF6UAaRZk3el2vtXhVPtvOn9uzW@dpg-cuue8c52ng1s739p7grg-a/data_suivi"
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

# Récupérer toutes les données de la table sensor_data
cursor.execute("SELECT * FROM sensor_data")
rows = cursor.fetchall()

# Récupérer les noms des colonnes
column_names = [desc[0] for desc in cursor.description]

# Écrire les données dans un fichier CSV
with open('sensor_data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(column_names)  # Écrire les en-têtes
    csvwriter.writerows(rows)  # Écrire toutes les lignes

# Convertir en DataFrame pandas si nécessaire
df = pd.DataFrame(rows, columns=column_names)
print(f"Données récupérées: {len(df)} lignes")

# Fermer la connexion
cursor.close()
conn.close()
# MLOps Project – Credit Card Fraud Detection

## 🎯 Objectif
Construire un pipeline complet MLOps pour détecter les transactions frauduleuses :
- Versioning des données (DVC)
- Tracking des expériences (MLflow)
- API de prédiction (FastAPI)
- Déploiement (Docker + CI/CD)
- Monitoring & drift detection (Prometheus, Evidently)

## 📊 Problème métier
Identifier automatiquement si une transaction bancaire est potentiellement frauduleuse.

## 🧪 Métriques principales
- AUC-ROC
- Recall (classe fraude)
- Precision-Recall AUC

## 🔍 Data Drift Monitoring
Ce projet utilise Evidently AI pour surveiller le dérive des données :

### Fonctionnalités
- **Feature Drift Detection** : Surveillance des changements dans les caractéristiques des données
- **Distribution Drift** : Détection des changements dans les distributions de données
- **Concept Drift** : Identification des changements dans les relations entre les features et la cible
- **Model Performance Monitoring** : Suivi des performances du modèle en production

### Composants
1. **Script de monitoring** : [src/monitoring/data_drift.py](file:///c:/Users/GIGABYTE/mlops-fraud/src/monitoring/data_drift.py)
2. **Rapports automatiques** : Générés quotidiennement ou à chaque nouvelle donnée
3. **Dashboard HTML** : Visualisation interactive des dérives détectées
4. **Export JSON** : Données structurées pour l'intégration système

### Utilisation
```bash
# Exécuter manuellement le monitoring
python src/monitoring/scheduled_monitoring.py

# Via l'API
curl -X POST http://localhost:8000/monitoring/drift/check
```

## 🤖 Model Drift Monitoring
Surveillance de la dégradation des performances du modèle en production :

### Fonctionnalités
- **Probability Anomaly Detection** : Détection des probabilités anormales
- **Prediction Drift** : Surveillance des changements dans les prédictions
- **Model Aging Detection** : Détection du vieillissement du modèle
- **Performance Comparison** : Comparaison des scores moyens avec les références

### Composants
1. **Script de monitoring** : [src/monitoring/model_drift.py](file:///c:/Users/GIGABYTE/mlops-fraud/src/monitoring/model_drift.py)
2. **Score de référence** : ROC-AUC calculé lors de l'évaluation du modèle
3. **Analyse des dernières prédictions** : Calcul des métriques sur les données de production
4. **Alertes automatiques** : Notifications en cas de dérive significative

### Utilisation
```bash
# Exécuter manuellement le monitoring
python src/monitoring/scheduled_model_monitoring.py

# Via l'API
curl -X POST http://localhost:8000/monitoring/model/check
```

## 📦 Stockage des requêtes en production
Pour monitorer le modèle en production, toutes les requêtes sont sauvegardées :

### Fonctionnalités
- **Stockage individuel** : Chaque requête est enregistrée dans un fichier séparé
- **Structure organisée** : Fichiers stockés dans `logs/requests/` avec horodatage
- **Format standardisé** : Chaque fichier contient timestamp, features, prédiction et probabilité

### Structure des logs
```
logs/
└── requests/
    ├── 20251208_103045_123456.json
    ├── 20251208_103046_789012.json
    └── ...
```

### Contenu d'un fichier de log
```json
{
    "timestamp": "2025-12-08T10:30:45.123456",
    "features": [0.1, 0.2, 0.3, ..., 1.0],
    "prediction": 1,
    "probability": 0.95
}
```

## 📂 Structure du projet
Voir l'arborescence dans la documentation.
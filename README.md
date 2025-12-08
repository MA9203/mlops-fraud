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

## 📂 Structure du projet
Voir l'arborescence dans la documentation.
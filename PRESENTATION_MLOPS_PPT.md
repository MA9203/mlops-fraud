# Présentation MLOps : Détection de fraude bancaire

---

## Slide 1 : Titre

# MLOps : Détection de fraude bancaire

### Nom : Arifa Mahjoub

### Encadrante : [Nom de ta prof]

### Date : [Date de présentation]

---

## Slide 2 : Contexte & Problématique

• Explosion des transactions électroniques
  - Croissance exponentielle des paiements digitaux
  - Volume de transactions sans précédent à traiter en temps réel

• Forte nécessité de détecter les fraudes rapidement
  - Coût économique élevé des fraudes bancaires
  - Impossibilité de contrôle humain à grande échelle

• Jeu de données : CreditCard Fraud Dataset (Kaggle)
  - Plus de 280 000 transactions
  - 30 features PCA anonymisées
  - Classe déséquilibrée (0.17% de fraudes)

• Objectif : construire un pipeline MLOps complet
  - Du data preprocessing → entraînement du modèle → API de prédiction → monitoring en production

---

## Slide 3 : Architecture Globale MLOps

```
Données Brutes → DVC → Entraînement (MLflow) → Packaging Docker 
                    ↓
API (FastAPI) → CI/CD → Railway → Monitoring (Evidently)
```

COMPOSANTS PRINCIPAUX :
• DVC : Versioning des données et pipeline reproductible
• MLflow : Suivi des expériences et versioning des modèles
• Docker : Conteneurisation pour déploiement universel
• FastAPI : API haute performance pour inférence
• GitHub Actions : Intégration et déploiement continus
• Railway : Plateforme de déploiement cloud
• Evidently : Monitoring de la qualité des données et modèles

---

## Slide 4 : Préparation des données

PRÉTRAITEMENT DES FEATURES :
• Normalisation de Amount : StandardScaler pour uniformiser l'échelle
• Anonymisation : Features PCA déjà transformées (V1-V28)
• Conservation : Time (timestamp) et Class (target)

GESTION DU DÉSÉQUILIBRE :
• Dataset fortement déséquilibré : 99.83% transactions légitimes
• Stratification utilisée lors du train/test split
• Maintien de la proportion de chaque classe dans les échantillons

PIPELINE REPRODUCTIBLE VIA DVC :
• Automatisation du preprocessing
• Traçabilité des transformations appliquées
• Versioning des datasets intermédiaires

---

## Slide 5 : Pipeline DVC

STAGES DU PIPELINE DVC :
1. preprocess : Transformation des données brutes
2. train : Entraînement du modèle
3. evaluate : Évaluation des performances

AVANTAGES DU PIPELINE DVC :
• Reproductibilité totale : Mêmes résultats à chaque exécution
• Traçabilité des fichiers : Suivi des dépendances et outputs
• Synchronisation Git : Intégration native avec le versioning de code
• Cache intelligent : Ré-exécution uniquement des étapes modifiées

VISUALISATION : Diagramme acyclique orienté montrant les dépendances

---

## Slide 6 : Entraînement & Suivi MLflow

MODÈLE MACHINE LEARNING :
• Algorithme : Régression Logistique (baseline)
• Features : 30 colonnes PCA + Amount normalisé
• Métriques : ROC AUC, Precision-Recall AUC
• Validation : Split 80/20 avec stratification

TRACKING MLFLOW :
• ROC AUC : Score de performance principale
• PR AUC : Performance sur classe minoritaire
• Paramètres : Hyperparamètres du modèle
• Modèle versionné : Sauvegarde dans le Model Registry

INTERFACE MLFLOW : Dashboard d'expérimentation avec comparaison des runs

---

## Slide 7 : API d'Inférence (FastAPI)

ENDPOINT PRINCIPAL : /predict
Réponse : { "prediction": 0, "probability": 0.15 }

FONCTIONNALITÉS CLÉS :
• Gestion des erreurs : Validation des inputs et codes d'erreur appropriés
• Tests unitaires PyTest : Couverture complète des endpoints
• Chargement automatique : Modèle chargé au démarrage de l'API
• Documentation auto-générée : Swagger UI intégré

---

## Slide 8 : Dockerisation

DOCKERFILE OPTIMISÉ :
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

AVANTAGES :
• Image légère : Base slim pour rapidité de déploiement
• Dépendances isolées : Environnement reproductible
• Portabilité : Fonctionne sur toute plateforme supportant Docker
• Scaling : Facile à déployer en cluster ou sur Kubernetes

---

## Slide 9 : CI/CD avec GitHub Actions

WORKFLOW AUTOMATISÉ :
1. Déclenchement : Push/Pull Request sur main/dev-clean
2. Testing : PyTest sur plusieurs versions Python
3. Qualité de code : Linting avec flake8
4. Build Docker : Construction des images
5. Déploiement : Envoi vers Railway

PIPELINE CI/CD :
Code Push → Tests → Build Docker → Deploy Railway

AVANTAGES :
• Validation automatique : Pas de déploiement sans tests
• Multi-environnement : Staging (dev-clean) et Production (main)
• Rollback facilité : Versions taggées dans Docker Hub

---

## Slide 10 : Déploiement Railway

ENDPOINTS PUBLICS :
• API : https://fraud-api-production.up.railway.app
• UI : https://fraud-ui-production.up.railway.app

HEALTH CHECK :
GET /health → { "status": "ok", "model_loaded": true, "n_features_expected": 30 }

PRÉDICTION EN LIGNE :
curl -X POST https://fraud-api-production.up.railway.app/predict

---

## Slide 11 : Monitoring (Semaine 7)

EVIDENTLY : DATA DRIFT DETECTION :
• Tests statistiques : Kolmogorov-Smirnov pour chaque feature
• Détection automatique : Alertes en cas de dérive significative
• Rapports : JSON et HTML pour analyse approfondie

DASHBOARD DE MONITORING :
Visualisation interactive des dérives de données

INTÉGRATIONS FUTURES :
• Cron jobs : Exécution quotidienne des analyses
• Ingestion automatique : Surveillance en continu
• Alerting : Notifications Slack/email en cas de problème

---

## Slide 12 : Conclusion & Améliorations Futures

PIPELINE COMPLET RÉALISÉ :
✓ Versioning des données (DVC)
✓ Suivi d'expériences (MLflow)
✓ API de prédiction (FastAPI)
✓ CI/CD automatisé (GitHub Actions)
✓ Déploiement cloud (Railway)
✓ Monitoring (Evidently)

AMÉLIORATIONS POSSIBLES :
• Model serving : Migration vers BentoML
• Base de données : PostgreSQL pour stockage transactionnel
• Feature Store : Implémentation de Feast
• Monitoring live : Intégration Grafana/Prometheus
• Modèles avancés : XGBoost, Random Forest, Deep Learning

IMPACT MÉTIER :
Solution prête pour la production avec traçabilité complète
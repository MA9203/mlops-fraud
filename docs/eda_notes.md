# 📊 Analyse Exploratoire des Données (EDA)
Dataset : Credit Card Fraud Detection

---

## 1. Informations générales
- Nombre de lignes : ~284 807
- Nombre de colonnes : 31
- Colonnes principales :
  - `Time`
  - `V1` à `V28` (composantes PCA)
  - `Amount`
  - `Class`

---

## 2. Qualité des données

### ✔ Valeurs manquantes
Aucune valeur manquante.

### ✔ Propreté
Dataset propre, colonnes déjà normalisées (PCA).

---

## 3. Déséquilibre des classes

- Classe 0 ≈ 99.83%
- Classe 1 ≈ 0.17%

➡️ Très fort déséquilibre.

---

## 4. Analyse des variables

- `V1`–`V28` : colonnes PCA.
- `Amount` : non normalisé → à scaler.
- `Time` : peut être transformé ou ignoré.

---

## 5. Corrélation

- Corrélations faibles (PCA oblige).
- Class faiblement corrélée à toutes les variables.

---

## 6. Implications pour le modèle

- Utiliser stratified split
- Utiliser metrics : Recall, AUC-ROC, AUC-PR
- Normaliser Amount
- Bien gérer le déséquilibre

---

## 7. Prochaine étape (Semaine 3)

- Script d’entraînement (`train.py`)
- MLflow tracking
- Premiers modèles (Logistic Regression, XGBoost)


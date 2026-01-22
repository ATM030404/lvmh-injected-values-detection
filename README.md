# Détection de valeurs injectées dans une série temporelle (LVMH)

Ce projet a pour objectif de détecter des **valeurs aberrantes injectées artificiellement** dans une série temporelle financière, en utilisant un modèle d’apprentissage automatique non supervisé (**Isolation Forest**).

Une **interface interactive Streamlit**, déployée en ligne, permet de tester facilement la méthode sans installation locale.

---

## Application en ligne

L’interface est accessible directement via un navigateur web :

 **https://lvmh-injected-values-detection-opzkz73genpnvhbcm7igq2.streamlit.app/**

Elle permet de :
- charger un fichier CSV contenant les colonnes `date` et `close`,
- ajuster les paramètres du modèle,
- lancer la détection des valeurs injectées,
- visualiser les résultats sous forme de tableau et de graphique,
- télécharger les valeurs détectées.

---

## Méthode utilisée

Le projet repose sur :
- un **modèle Isolation Forest** entraîné sur les données,
- une ingénierie de variables incluant :
  - log-prix,
  - log-retours,
  - statistiques glissantes (moyenne et écart-type),
  - z-score dynamique,
- un filtrage local permettant de distinguer les véritables valeurs injectées des variations normales du marché.

Cette approche permet de détecter efficacement des anomalies ponctuelles, indépendamment des tendances longues.

---





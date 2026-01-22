import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Détection valeurs injectées (LVMH)", layout="wide")

st.title("Détection de valeurs injectées dans une série temporelle (LVMH)")
st.write("Upload un CSV avec colonnes `date` et `close`, puis lance la détection.")

# =======================
# Sidebar paramètres
# =======================
st.sidebar.header("Paramètres")
fenetre_rolling = st.sidebar.number_input("Fenêtre rolling (jours)", min_value=5, max_value=120, value=20, step=1)
contamination = st.sidebar.number_input("Contamination (taux d’anomalies)", min_value=0.0005, max_value=0.02, value=0.002, step=0.0005, format="%.4f")
nb_arbres = st.sidebar.number_input("Nombre d’arbres", min_value=50, max_value=2000, value=500, step=50)
graine = st.sidebar.number_input("Graine aléatoire", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.subheader("Filtre injection")
seuil_ratio_voisins = st.sidebar.number_input("Seuil ratio voisins (R)", min_value=1.5, max_value=10.0, value=3.0, step=0.1)
seuil_zinj = st.sidebar.number_input("Seuil |zscore| (Z_INJ)", min_value=2.0, max_value=10.0, value=3.5, step=0.1)
jours_proches = st.sidebar.number_input("Fenêtre anti-rebond (jours)", min_value=1, max_value=10, value=3, step=1)

# =======================
# Upload fichier
# =======================
fichier = st.file_uploader("Choisis ton fichier CSV", type=["csv"])

if fichier is None:
    st.info("Upload un CSV pour commencer.")
    st.stop()

sep = st.selectbox("Séparateur CSV", options=[";", ",", "\t"], index=0)

try:
    donnees = pd.read_csv(fichier, sep=sep)
except Exception as e:
    st.error(f"Impossible de lire le CSV : {e}")
    st.stop()

# Normaliser les noms de colonnes
donnees.columns = [c.lower().strip() for c in donnees.columns]

if "date" not in donnees.columns or "close" not in donnees.columns:
    st.error("Le fichier doit contenir les colonnes `date` et `close`.")
    st.write("Colonnes détectées :", list(donnees.columns))
    st.stop()

# =======================
# Bouton exécution
# =======================
lancer = st.button("Lancer la détection", type="primary")
if not lancer:
    st.stop()

# =======================
# Pipeline
# =======================
try:
    # Date
    donnees["date"] = pd.to_datetime(donnees["date"], dayfirst=True, errors="coerce")

    # Close -> numérique (gère virgules éventuelles)
    donnees["close"] = (
        donnees["close"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    donnees["close"] = pd.to_numeric(donnees["close"], errors="coerce")

    donnees = donnees.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # Log-returns
    donnees["log_return"] = np.log(donnees["close"]).diff()
    donnees = donnees.dropna(subset=["log_return"]).reset_index(drop=True)

    # Rolling + zscore
    donnees["roll_mean"] = donnees["log_return"].rolling(fenetre_rolling).mean()
    donnees["roll_std"]  = donnees["log_return"].rolling(fenetre_rolling).std()
    donnees["zscore"] = (donnees["log_return"] - donnees["roll_mean"]) / donnees["roll_std"]
    donnees = donnees.dropna(subset=["roll_mean", "roll_std", "zscore"]).reset_index(drop=True)

    # Log-price
    donnees["log_price"] = np.log(donnees["close"])

    # Features + normalisation
    X = donnees[["log_price", "log_return", "roll_mean", "roll_std", "zscore"]].values
    Xn = StandardScaler().fit_transform(X)

    # Modèle
    modele = IsolationForest(
        n_estimators=int(nb_arbres),
        contamination=float(contamination),
        random_state=int(graine)
    ).fit(Xn)

    donnees["anomaly_score"] = -modele.score_samples(Xn)
    donnees["is_anomaly"] = (modele.predict(Xn) == -1).astype(int)

    # Filtre injection
    neighbor_mean = (donnees["close"].shift(1) + donnees["close"].shift(-1)) / 2
    neighbor_mean = neighbor_mean.replace(0, np.nan)
    ratio_to_neighbors = donnees["close"] / neighbor_mean

    is_injected_candidate = (
        (ratio_to_neighbors > seuil_ratio_voisins)
        | (ratio_to_neighbors < 1 / seuil_ratio_voisins)
        | (donnees["zscore"].abs() > seuil_zinj)
    )

    injectees = donnees[(donnees["is_anomaly"] == 1) & (is_injected_candidate)].copy()
    injectees["ratio_to_neighbors"] = ratio_to_neighbors.loc[injectees.index].values

    # Anti-rebond
    injectees = injectees.sort_values("date").reset_index(drop=True)
    to_drop = set()

    for i in range(1, len(injectees)):
        d0 = injectees.loc[i - 1, "date"]
        d1 = injectees.loc[i, "date"]
        if (d1 - d0).days <= int(jours_proches):
            e0 = abs(np.log(injectees.loc[i - 1, "ratio_to_neighbors"]))
            e1 = abs(np.log(injectees.loc[i, "ratio_to_neighbors"]))
            to_drop.add(i if e0 >= e1 else i - 1)

    injectees = injectees.drop(index=to_drop).sort_values("date").reset_index(drop=True)

except Exception as e:
    st.error("Erreur pendant le traitement. Vérifie le format du fichier (date, close) et le séparateur.")
    st.exception(e)
    st.stop()

# =======================
# Résultats
# =======================
st.subheader("Résultats")
st.write(f"**Nombre total de valeurs injectées détectées : {len(injectees)}**")

colonnes_aff = ["date", "close", "ratio_to_neighbors", "log_return", "zscore", "anomaly_score"]
st.dataframe(injectees[colonnes_aff], use_container_width=True)

csv_out = injectees[colonnes_aff].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Télécharger les valeurs injectées (CSV)",
    data=csv_out,
    file_name="valeurs_injectees_probables.csv",
    mime="text/csv",
)

# Graph
st.subheader("Graphique")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(donnees["date"], donnees["close"], label="Close")
ax.scatter(injectees["date"], injectees["close"], marker="x", label="Valeurs injectées probables")
ax.set_title("Détection des valeurs injectées")
ax.set_xlabel("Date")
ax.set_ylabel("Prix de clôture")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

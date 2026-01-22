import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Détection valeurs injectées (LVMH)", layout="wide")

st.title("Détection de valeurs injectées dans une série temporelle (LVMH)")
st.write("Upload un CSV avec colonnes `date` et `close`, puis lance la détection sur 3 modèles.")


# =======================
# Sidebar paramètres
# =======================
st.sidebar.header("Paramètres communs")
fenetre_rolling = st.sidebar.number_input("Fenêtre rolling (jours)", min_value=5, max_value=120, value=20, step=1)
contamination = st.sidebar.number_input(
    "Contamination (taux d’anomalies)", min_value=0.0005, max_value=0.02, value=0.002, step=0.0005, format="%.4f"
)
graine = st.sidebar.number_input("Graine aléatoire", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.subheader("Filtre injection")
seuil_ratio_voisins = st.sidebar.number_input("Seuil ratio voisins (R)", min_value=1.5, max_value=10.0, value=3.0, step=0.1)
seuil_zinj = st.sidebar.number_input("Seuil |zscore| (Z_INJ)", min_value=2.0, max_value=10.0, value=3.5, step=0.1)
jours_proches = st.sidebar.number_input("Fenêtre anti-rebond (jours)", min_value=1, max_value=10, value=3, step=1)

st.sidebar.header("Paramètres modèles")
nb_arbres = st.sidebar.number_input("Isolation Forest : arbres", min_value=50, max_value=2000, value=500, step=50)
lof_voisins = st.sidebar.number_input("LOF : voisins", min_value=5, max_value=200, value=20, step=1)
ocsvm_kernel = st.sidebar.selectbox("One-Class SVM : kernel", ["rbf", "poly", "sigmoid", "linear"], index=0)
ocsvm_gamma = st.sidebar.selectbox("One-Class SVM : gamma", ["scale", "auto"], index=0)

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

# normalise les noms de colonnes
donnees.columns = [c.lower().strip() for c in donnees.columns]

if "date" not in donnees.columns or "close" not in donnees.columns:
    st.error("Le fichier doit contenir les colonnes `date` et `close`.")
    st.write("Colonnes détectées :", list(donnees.columns))
    st.stop()

lancer = st.button("Lancer la détection (3 modèles)", type="primary")
if not lancer:
    st.stop()

# =======================
# Pipeline commun : nettoyage + features
# =======================
donnees["date"] = pd.to_datetime(donnees["date"], dayfirst=True, errors="coerce")
donnees = donnees.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

# RETURN simple 
donnees["return"] = (donnees["close"].pct_change() * 100).round(2)


# Log-return
donnees["log_return"] = np.log(donnees["close"]).diff()

donnees = donnees.dropna(subset=["return", "log_return"]).reset_index(drop=True)

donnees["roll_mean"] = donnees["log_return"].rolling(int(fenetre_rolling)).mean()
donnees["roll_std"] = donnees["log_return"].rolling(int(fenetre_rolling)).std()
donnees["zscore"] = (donnees["log_return"] - donnees["roll_mean"]) / donnees["roll_std"]
donnees = donnees.dropna(subset=["roll_mean", "roll_std", "zscore"]).reset_index(drop=True)

donnees["log_price"] = np.log(donnees["close"])

features = ["log_price", "log_return", "roll_mean", "roll_std", "zscore"]
X = donnees[features].values
Xn = StandardScaler().fit_transform(X)

neighbor_mean = (donnees["close"].shift(1) + donnees["close"].shift(-1)) / 2
neighbor_mean = neighbor_mean.replace(0, np.nan)
ratio_to_neighbors = donnees["close"] / neighbor_mean

def filtrer_injections(df_local, score_col, is_anomaly_col):
    df2 = df_local.copy()
    df2["ratio_to_neighbors"] = ratio_to_neighbors.values

    is_injected_candidate = (
        (df2["ratio_to_neighbors"] > float(seuil_ratio_voisins))
        | (df2["ratio_to_neighbors"] < 1.0 / float(seuil_ratio_voisins))
        | (df2["zscore"].abs() > float(seuil_zinj))
    )

    injectees = df2[(df2[is_anomaly_col] == 1) & (is_injected_candidate)].copy()

    # anti-rebond (garder la plus extrême si dates proches)
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
    return injectees

# =======================
# 1) Isolation Forest
# =======================
df_if = donnees.copy()

model_if = IsolationForest(
    n_estimators=int(nb_arbres),
    contamination=float(contamination),
    random_state=int(graine)
).fit(Xn)

df_if["anomaly_score"] = -model_if.score_samples(Xn)
df_if["is_anomaly"] = (model_if.predict(Xn) == -1).astype(int)

injectees_if = filtrer_injections(df_if, "anomaly_score", "is_anomaly")

# =======================
# 2) LOF
# =======================
df_lof = donnees.copy()

lof = LocalOutlierFactor(
    n_neighbors=int(lof_voisins),
    contamination=float(contamination)
)

pred_lof = lof.fit_predict(Xn)
df_lof["is_anomaly"] = (pred_lof == -1).astype(int)
df_lof["anomaly_score"] = -lof.negative_outlier_factor_

injectees_lof = filtrer_injections(df_lof, "anomaly_score", "is_anomaly")

# =======================
# 3) One-Class SVM
# =======================
df_svm = donnees.copy()

svm = OneClassSVM(
    kernel=ocsvm_kernel,
    nu=float(contamination),
    gamma=ocsvm_gamma
).fit(Xn)

df_svm["anomaly_score"] = -svm.decision_function(Xn)
df_svm["is_anomaly"] = (svm.predict(Xn) == -1).astype(int)

injectees_svm = filtrer_injections(df_svm, "anomaly_score", "is_anomaly")

# =======================
# Affichage résultats
# =======================
st.subheader("Résultats par modèle")

tab1, tab2, tab3 = st.tabs(["Isolation Forest (recommandé)", "LOF", "One-Class SVM"])

def afficher_modele(nom, df_all, inj):
    st.write(f"**Nombre total de valeurs injectées détectées : {len(inj)}**")

    cols = [
        "date",
        "close",
        "return",
        "ratio_to_neighbors",
        "log_return",
        "zscore",
        "anomaly_score"
    ]

    st.dataframe(inj[cols], use_container_width=True)

    csv_out = inj[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Télécharger (CSV) - {nom}",
        data=csv_out,
        file_name=f"valeurs_injectees_{nom.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    fig = plt.figure(figsize=(12, 4))
    plt.plot(df_all["date"], df_all["close"], label="Close")
    plt.scatter(inj["date"], inj["close"], marker="x", label="Valeurs injectées probables")
    plt.title(f"Détection des valeurs injectées – {nom}")
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab1:
    afficher_modele("Isolation Forest", df_if, injectees_if)

with tab2:
    afficher_modele("LOF", df_lof, injectees_lof)

with tab3:
    afficher_modele("One-Class SVM", df_svm, injectees_svm)

st.success("Conclusion : le meilleur modèle ici est **Isolation Forest** (stabilité + cohérence des anomalies).")

"""Interface Streamlit (en francais) : IDS hybride SVM + K-Means sur NSL-KDD.

Usage : streamlit run app.py
Page unique : formulaire (10 champs) + un seul bouton Analyser,
puis verdicts et performances. Menu Streamlit masque pour une demo jury sobre.
Analyse de connexions decrites par les features NSL-KDD uniquement :
aucune capture ni analyse temps reel du reseau local.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

COLUMNS_41 = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

st.set_page_config(page_title="IDS hybride SVM + K-Means", layout="centered")
st.markdown(
    "<style>#MainMenu, header, footer, "
    ".stAppDeployButton, .stToolbar {display: none;}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    # Securite : joblib charge uniquement nos propres fichiers models/*.pkl
    # generes localement par train.py (source de confiance), jamais un fichier distant.
    paths = {k: os.path.join(MODELS_DIR, f)
             for k, f in {"scaler": "scaler.pkl", "encoder": "encoder.pkl",
                          "svm": "svm.pkl", "kmeans": "kmeans.pkl",
                          "hybrid": "svm_hybrid.pkl",
                          "meta": "meta.json"}.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    bundle = {k: joblib.load(p) for k, p in paths.items() if k != "meta"}
    bundle["meta"] = json.load(open(paths["meta"], encoding="utf-8"))
    return bundle


def vectorize(df_raw, bundle):
    """Applique le meme pipeline que train.py : OneHot + StandardScaler."""
    meta = bundle["meta"]
    num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
    for c in num_cols:
        if c not in df_raw:
            df_raw[c] = meta["defaults"].get(c, 0.0)
    for c in cat_cols:
        if c not in df_raw:
            df_raw[c] = meta["defaults"].get(c, "")
    df_raw = df_raw.copy()
    df_raw[num_cols] = df_raw[num_cols].fillna(0).astype(float)
    X_cat = bundle["encoder"].transform(df_raw[cat_cols].astype(str))
    X_full = np.hstack([df_raw[num_cols].astype(float).values, X_cat])
    return bundle["scaler"].transform(X_full)


def predict_one(row, bundle):
    Xs = vectorize(pd.DataFrame([row]), bundle)
    pred_svm = int(bundle["svm"].predict(Xs)[0])
    d = bundle["kmeans"].transform(Xs)
    pred_h = int(bundle["hybrid"].predict(np.hstack([Xs, d]))[0])
    cluster = int(bundle["kmeans"].predict(Xs)[0])
    return pred_svm, pred_h, cluster


st.title("IDS hybride : SVM + K-Means (NSL-KDD)")
st.write("Décrivez une connexion (10 champs clés, le reste = médianes du "
         "train), puis cliquez sur Analyser. Dataset uniquement, pas de "
         "capture réseau en temps réel.")

bundle = load_models()
if bundle is None:
    st.warning("Modèles non trouvés. Lancez d'abord : python train.py.")
    st.stop()
meta = bundle["meta"]
dflt = meta["defaults"]

c1, c2, c3 = st.columns(3)
with c1:
    duration = st.number_input("duration (s)", min_value=0.0,
                               value=float(dflt.get("duration", 0)))
    protocol_type = st.selectbox("protocol_type", ["tcp", "udp", "icmp"],
                                 index=["tcp", "udp", "icmp"].index(
                                     str(dflt.get("protocol_type", "tcp"))))
    service = st.text_input("service", value=str(dflt.get("service", "http")))
    flag = st.selectbox("flag", ["SF", "S0", "REJ", "RSTR", "SH", "OTH"], index=0)
with c2:
    src_bytes = st.number_input("src_bytes", min_value=0.0,
                                value=float(dflt.get("src_bytes", 0)))
    dst_bytes = st.number_input("dst_bytes", min_value=0.0,
                                value=float(dflt.get("dst_bytes", 0)))
    logged_in = st.selectbox("logged_in", [0, 1], index=int(dflt.get("logged_in", 0)))
    count = st.number_input("count", min_value=0.0, value=float(dflt.get("count", 0)))
with c3:
    srv_count = st.number_input("srv_count", min_value=0.0,
                                value=float(dflt.get("srv_count", 0)))
    dst_host_count = st.number_input("dst_host_count", min_value=0.0,
                                     value=float(dflt.get("dst_host_count", 0)))

if st.button("Analyser", type="primary"):
    row = {c: dflt.get(c, 0.0) for c in COLUMNS_41}
    row.update({"duration": duration, "protocol_type": protocol_type,
                "service": service, "flag": flag, "src_bytes": src_bytes,
                "dst_bytes": dst_bytes, "logged_in": logged_in,
                "count": count, "srv_count": srv_count,
                "dst_host_count": dst_host_count})
    pred_svm, pred_h, cluster = predict_one(row, bundle)
    label = lambda p: "Intrusion suspectée" if p == 1 else "Trafic normal probable"
    st.subheader("Résultat")
    st.write(f"SVM seul : **{label(pred_svm)}**")
    st.write(f"Hybride : **{label(pred_h)}** (cluster K-Means n°{cluster})")
    if pred_svm != pred_h:
        st.info("Les deux modèles divergent : l'info de regroupement K-Means "
                "a fait basculer la décision — cas intéressant pour le jury.")

st.divider()
st.subheader("Performances sur KDDTest+ complet (22 544 connexions)")
m_svm, m_h = meta["metrics_svm"], meta["metrics_hybrid"]
st.write(f"SVM seul : accuracy {m_svm['accuracy']}, précision {m_svm['precision']}, "
         f"rappel {m_svm['recall']}, F1 {m_svm['f1']}, FPR {m_svm['fpr']}")
st.write(f"Hybride : accuracy {m_h['accuracy']}, précision {m_h['precision']}, "
         f"rappel {m_h['recall']}, F1 {m_h['f1']}, FPR {m_h['fpr']}")
st.caption("Évaluation honnête : attaques inédites du test jamais vues à "
           "l'entraînement ; l'hybride perd en rappel ici (voir RAPPORT.md).")

metrics = ["precision", "recall", "f1", "accuracy"]
fig, ax = plt.subplots()
x = range(len(metrics))
ax.bar([i - 0.2 for i in x], [m_svm[m] for m in metrics], width=0.4, label="SVM")
ax.bar([i + 0.2 for i in x], [m_h[m] for m in metrics], width=0.4, label="Hybride")
ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()
st.pyplot(fig)
st.caption(f"Matrices de confusion [TN FP / FN TP] — SVM : {meta['confusion_svm']} ; "
           f"hybride : {meta['confusion_hybrid']}.")

"""IDS hybride SVM + K-Means sur NSL-KDD (detection binaire normal / intrusion).

Usage : python train.py [--sample N] [--kernel rbf|linear] [--kmeans K]
Attend data/KDDTrain+.txt et data/KDDTest+.txt (fournis par l'etudiant).

Protocole :
- Train : KDDTrain+ (echantillon stratifie N=25000 par defaut, seed 42,
  car un SVM RBF sur 125k lignes est trop couteux pour un TP).
- Test : KDDTest+ COMPLET (22544 lignes, protocole officiel, avec attaques inedites).
- Pas de fuite : OneHotEncoder, StandardScaler et K-Means ajustes sur train uniquement.
- Hybride (stacking, meme methode que le projet phishing-detector) :
  distances aux K centres K-Means concatenees aux features, puis SVM memes hyperparametres.
"""
import argparse
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "data", "KDDTrain+.txt")
TEST_PATH = os.path.join(BASE, "data", "KDDTest+.txt")
MODELS_DIR = os.path.join(BASE, "models")
OUTPUTS_DIR = os.path.join(BASE, "outputs")

# 41 features NSL-KDD + etiquette + difficulte (fichiers .txt sans en-tete).
COLUMNS = [
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
    "label", "difficulty",
]
CAT_COLS = ["protocol_type", "service", "flag"]

# Les 10 champs exposes dans le formulaire simplifie de app.py.
DEMO_FIELDS = ["duration", "protocol_type", "service", "flag", "src_bytes",
               "dst_bytes", "logged_in", "count", "srv_count", "dst_host_count"]


def load_nslkdd(path):
    """Charge un fichier NSL-KDD .txt (sans en-tete) avec les noms de colonnes."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def to_binary(series):
    """normal -> 0, tout autre type d'attaque -> 1."""
    return (series.astype(str).str.strip().str.lower() != "normal").astype(int)


def build_preprocessor(df_train):
    """Ajuste OneHot (categorieles) + StandardScaler (tout) sur train uniquement."""
    num_cols = [c for c in COLUMNS if c not in CAT_COLS + ["label", "difficulty"]]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit(df_train[CAT_COLS].astype(str))
    X_cat = encoder.transform(df_train[CAT_COLS].astype(str))
    try:
        cat_names = list(encoder.get_feature_names_out(CAT_COLS))
    except AttributeError:
        cat_names = [f"{c}_{v}" for c in CAT_COLS
                     for v in encoder.categories_[CAT_COLS.index(c)]]
    X_full = np.hstack([df_train[num_cols].fillna(0).astype(float).values, X_cat])
    feature_names = num_cols + cat_names
    scaler = StandardScaler()
    scaler.fit(X_full)
    return encoder, scaler, num_cols, cat_names, feature_names


def transform(df, encoder, scaler, num_cols):
    X_cat = encoder.transform(df[CAT_COLS].astype(str))
    X_full = np.hstack([df[num_cols].fillna(0).astype(float).values, X_cat])
    return scaler.transform(X_full)


def scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(fpr, 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Entraine l'IDS hybride SVM + K-Means.")
    ap.add_argument("--sample", type=int, default=25000,
                    help="Taille de l'echantillon stratifie de KDDTrain+ (defaut 25000).")
    ap.add_argument("--kernel", default="rbf", choices=["rbf", "linear"])
    ap.add_argument("--kmeans", type=int, default=2, help="Nombre de clusters K-Means.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in (TRAIN_PATH, TEST_PATH):
        if not os.path.exists(p):
            print(f"[ERREUR] Fichier manquant : {p}")
            print("  Placez KDDTrain+.txt et KDDTest+.txt dans data/ puis relancez.")
            sys.exit(1)

    t0 = time.time()
    df_train = load_nslkdd(TRAIN_PATH)
    df_test = load_nslkdd(TEST_PATH)
    print(f"[INFO] Train : {df_train.shape[0]} lignes, Test : {df_test.shape[0]} lignes")
    print("[INFO] Attaques distinctes (train) :",
          sorted(df_train["label"].astype(str).str.strip().unique()))
    y_train_full = to_binary(df_train["label"])
    y_test = to_binary(df_test["label"])
    print("[INFO] Repartition train (0=normal, 1=intrusion) :")
    print(y_train_full.value_counts().to_string())
    print("[INFO] Repartition test (0=normal, 1=intrusion) :")
    print(y_test.value_counts().to_string())

    # Echantillon stratifie reproductible pour le SVM RBF.
    n_sample = min(args.sample, len(df_train))
    idx, y_samp = train_test_split(
        np.arange(len(df_train)), train_size=n_sample, random_state=args.seed,
        stratify=y_train_full)
    df_tr = df_train.iloc[np.sort(idx)].reset_index(drop=True)
    y_tr = y_train_full.iloc[np.sort(idx)].values
    print(f"[INFO] Echantillon d'entrainement : {len(df_tr)} lignes "
          f"(stratifié, seed={args.seed})")

    encoder, scaler, num_cols, cat_names, feature_names = build_preprocessor(df_tr)
    print(f"[INFO] Features : {len(num_cols)} numeriques + {len(cat_names)} "
          f"indicatrices = {len(feature_names)} colonnes")
    X_tr = transform(df_tr, encoder, scaler, num_cols)
    X_te = transform(df_test, encoder, scaler, num_cols)

    # --- SVM seul ---
    print(f"[INFO] Entrainement SVM ({args.kernel}) ...")
    svm = SVC(kernel=args.kernel, random_state=args.seed)
    svm.fit(X_tr, y_tr)
    pred_svm = svm.predict(X_te)

    # --- K-Means non supervise (jamais d'etiquettes) ---
    print(f"[INFO] Entrainement K-Means (k={args.kmeans}) ...")
    kmeans = KMeans(n_clusters=args.kmeans, random_state=args.seed, n_init=10)
    kmeans.fit(X_tr)

    # --- Hybride : distances aux centres comme 2 features supplementaires ---
    d_tr = kmeans.transform(X_tr)
    d_te = kmeans.transform(X_te)
    svm_h = SVC(kernel=args.kernel, random_state=args.seed)
    svm_h.fit(np.hstack([X_tr, d_tr]), y_tr)
    pred_h = svm_h.predict(np.hstack([X_te, d_te]))

    m_svm = scores(y_test, pred_svm)
    m_h = scores(y_test, pred_h)
    cm_svm = confusion_matrix(y_test, pred_svm).tolist()
    cm_h = confusion_matrix(y_test, pred_h).tolist()
    print("\n=== SVM seul ===")
    print(m_svm)
    print("Matrice de confusion [TN FP / FN TP] :", cm_svm)
    print("\n=== Hybride K-Means + SVM ===")
    print(m_h)
    print("Matrice de confusion [TN FP / FN TP] :", cm_h)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(svm_h, os.path.join(MODELS_DIR, "svm_hybrid.pkl"))

    # Valeurs par defaut (medianes/modes du train) pour le formulaire simplifie.
    defaults = {}
    for c in num_cols:
        defaults[c] = float(df_tr[c].median())
    for c in CAT_COLS:
        defaults[c] = str(df_tr[c].mode().iloc[0])
    meta = {
        "kernel": args.kernel, "kmeans_k": args.kmeans, "seed": args.seed,
        "n_train_full": int(len(df_train)), "n_train_sample": int(len(df_tr)),
        "n_test": int(len(df_test)),
        "num_cols": num_cols, "cat_cols": CAT_COLS,
        "cat_feature_names": cat_names, "feature_names": feature_names,
        "demo_fields": DEMO_FIELDS, "defaults": defaults,
        "metrics_svm": m_svm, "metrics_hybrid": m_h,
        "confusion_svm": cm_svm, "confusion_hybrid": cm_h,
        "train_attacks": sorted(df_train["label"].astype(str).str.strip().unique().tolist()),
        "test_attacks": sorted(df_test["label"].astype(str).str.strip().unique().tolist()),
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- Graphiques jury (PNG) ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [m_svm[m] for m in labels], width=0.4, label="SVM seul")
    ax.bar(x + 0.2, [m_h[m] for m in labels], width=0.4, label="Hybride")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("SVM seul vs Hybride (KDDTest+)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "comparaison_metriques.png"), dpi=120)
    plt.close(fig)

    for name, cm in (("svm", cm_svm), ("hybride", cm_h)):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["normal", "intrusion"],
                    yticklabels=["normal", "intrusion"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reel")
        ax.set_title(f"Matrice de confusion — {name} (KDDTest+)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUTS_DIR, f"matrice_confusion_{name}.png"), dpi=120)
        plt.close(fig)

    print(f"\n[OK] Modeles dans {MODELS_DIR}/, graphiques dans {OUTPUTS_DIR}/ "
          f"({round(time.time() - t0, 1)} s au total)")


if __name__ == "__main__":
    main()

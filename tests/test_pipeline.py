"""Tests de verification de l'IDS (a lancer avec le .venv du projet).

Usage : .venv/bin/python tests/test_pipeline.py
Verifie : presence des artefacts, coherence des metriques avec meta.json,
fonctionnement de la prediction sur des lignes reelles de KDDTest+.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from train import COLUMNS, transform, to_binary  # noqa: E402

MODELS_DIR = os.path.join(BASE, "models")
TEST_PATH = os.path.join(BASE, "data", "KDDTest+.txt")

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), "-", name, detail)
    if not cond:
        FAILURES.append(name)


def main():
    for f in ["scaler.pkl", "encoder.pkl", "svm.pkl", "kmeans.pkl",
              "svm_hybrid.pkl", "meta.json"]:
        check(f"artefact {f}", os.path.exists(os.path.join(MODELS_DIR, f)))
    meta = json.load(open(os.path.join(MODELS_DIR, "meta.json"), encoding="utf-8"))
    # Securite : chargement joblib limite a nos propres modeles generes par train.py.
    bundle = {k: joblib.load(os.path.join(MODELS_DIR, k + ".pkl"))
              for k in ["scaler", "encoder", "svm", "kmeans", "svm_hybrid"]}

    df = pd.read_csv(TEST_PATH, header=None, names=COLUMNS, nrows=2000)
    y = to_binary(df["label"]).values
    Xs = transform(df, bundle["encoder"], bundle["scaler"], meta["num_cols"])
    check("dimensions transform", Xs.shape == (2000, len(meta["feature_names"])),
          f"obtenu {Xs.shape}")
    check("pas de NaN apres transform", bool(np.isfinite(Xs).all()))

    pred_svm = bundle["svm"].predict(Xs)
    d = bundle["kmeans"].transform(Xs)
    pred_h = bundle["svm_hybrid"].predict(np.hstack([Xs, d]))
    check("pred SVM binaire", set(np.unique(pred_svm)) <= {0, 1})
    check("pred hybride binaire", set(np.unique(pred_h)) <= {0, 1})
    check("clusters kmeans dans [0, k[",
          bool((bundle["kmeans"].predict(Xs) < meta["kmeans_k"]).all()))

    # Coherence : l'accuracy sur 2000 lignes doit etre proche de celle de meta.json.
    acc_svm = float((pred_svm == y).mean())
    check("accuracy SVM echantillon coherente avec meta.json",
          abs(acc_svm - meta["metrics_svm"]["accuracy"]) < 0.05,
          f"echantillon={acc_svm:.4f} vs meta={meta['metrics_svm']['accuracy']}")

    # Ligne unique (cas app.py) : ne doit pas lever d'exception.
    one = transform(df.iloc[[0]], bundle["encoder"], bundle["scaler"],
                    meta["num_cols"])
    try:
        bundle["svm"].predict(one)
        bundle["svm_hybrid"].predict(
            np.hstack([one, bundle["kmeans"].transform(one)]))
        check("prediction ligne unique", True)
    except Exception as e:  # noqa: BLE001
        check("prediction ligne unique", False, str(e))

    print()
    if FAILURES:
        print(f"[ECHEC] {len(FAILURES)} test(s) en echec : {FAILURES}")
        sys.exit(1)
    print("[OK] Tous les tests passent.")


if __name__ == "__main__":
    main()

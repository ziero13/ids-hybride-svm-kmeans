# IDS hybride pour la détection d'intrusions réseau (SVM + K-Means, NSL-KDD)

TP universitaire — cybersécurité & Machine Learning. Analyse de trafic réseau
décrite par le dataset NSL-KDD : classifier les connexions (normal vs intrusion)
avec un SVM, regrouper avec K-Means, combiner les deux (distances aux centres
K-Means ajoutées comme features pour le SVM) — même méthode hybride que le
projet `phishing-detector` (K-NN + K-Means).

> Portée honnête : analyse d'un dataset en laboratoire uniquement. Ce projet
> ne capture pas le réseau et ne détecte rien en temps réel.

## Structure

```
ids-hybride-svm-kmeans/
├── data/
│   ├── KDDTrain+.txt   # 125 973 connexions (fourni par l'étudiant)
│   └── KDDTest+.txt    # 22 544 connexions (fourni par l'étudiant)
├── models/             # modèles entraînés (générés par train.py)
├── outputs/            # graphiques jury (générés par train.py)
├── tests/              # tests de vérification
├── variants/k5/        # métrics + graphiques de la variante K-Means k=5
├── app.py              # interface Streamlit (français)
├── train.py            # prétraitement, entraînement, évaluation
├── requirements.txt
└── README.md / RAPPORT.md
```

## Données (vérifiées, jamais inventées)

- **Source** : archive NSL-KDD fournie par l'étudiant (`KDDTrain+.txt`,
  `KDDTest+.txt`, format sans en-tête : 41 features + label + difficulty).
- **Train** : 125 973 lignes — 67 343 normal (53,5 %) / 58 630 attaques,
  23 modalités (neptune 41 214 … spy 2, perl 3).
- **Test (KDDTest+ natif)** : 22 544 lignes — 9 711 normal (43 %) /
  12 833 attaques (57 %), avec ~17 types d'attaques **inédites**
  (mscan, apache2, saint, httptunnel…).
- **Binaire** : `normal` → 0, tout autre label → 1.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

```bash
python train.py [--sample 25000] [--kernel rbf] [--kmeans 2] [--seed 42]
streamlit run app.py
.venv/bin/python tests/test_pipeline.py
```

`train.py` échantillonne KDDTrain+ (stratifié, seed 42, défaut 25 000 lignes :
un SVM RBF sur 125k lignes est trop coûteux pour un TP) et évalue sur
**KDDTest+ complet** — protocole officiel, attaques inédites incluses.
OneHotEncoder, StandardScaler et K-Means sont ajustés **sur le train
uniquement** (pas de fuite).

## Algorithmes

- **SVM (RBF)** : classifieur supervisé sur 117 colonnes (38 numériques
  standardisées + 79 indicatrices protocol_type/service/flag).
- **K-Means (k=2)** : clustering non supervisé, jamais d'étiquettes.
- **Hybride (stacking)** : distances aux 2 centres concaténées aux features,
  puis SVM mêmes hyperparamètres. Comparaison sur les mêmes données de test.

## Résultats (KDDTest+ complet, 22 544 connexions — sortie réelle de train.py)

| Modèle | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| SVM seul | 0.7934 | 0.9249 | 0.6934 | 0.7926 | 0.0743 |
| Hybride k=2 | 0.7622 | 0.9205 | 0.6372 | 0.7531 | 0.0727 |
| Hybride k=5 (variante) | 0.7600 | 0.9191 | 0.6341 | 0.7505 | 0.0737 |

Confusions [TN FP / FN TP] : SVM `[[8989, 722], [3935, 8898]]`,
hybride k=2 `[[9005, 706], [4656, 8177]]`.

**Lecture honnête** : ici l'hybride stacking ne bat pas le SVM seul —
il perd en rappel (−5,6 pts : 721 intrusions manquées de plus, dont des
attaques inédites du test proches du cluster « normal »). Le FPR baisse
légèrement (0,0743 → 0,0727, 16 faux positifs évités). Détail dans RAPPORT.md.

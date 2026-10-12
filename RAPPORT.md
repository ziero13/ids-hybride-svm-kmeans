# Rapport — IDS hybride SVM + K-Means sur NSL-KDD

## 1. Dataset : provenance, format, répartition (vérifiés)

- **Provenance** : archive NSL-KDD fournie par l'étudiant (fichiers datés
  2019-10-22 : `KDDTrain+.txt`, `KDDTest+.txt`, variantes 20 % et fichiers
  `.arff`/`.jpg` non utilisés). Contrôles effectués : 125 973 / 22 544 lignes
  (`wc -l`), 43 champs par ligne, labels listés par comptage direct.
- **Format** : texte sans en-tête, 41 features + label (ex. `normal`,
  `neptune`, `satan`, `mscan`…) + score de difficulté. Noms de colonnes
  appliqués d'après la documentation NSL-KDD (colonne 42 = label).
- **Répartition** : train 67 343 normal (53,5 %) / 58 630 attaques (23
  modalités, de `neptune` 41 214 à `spy` 2) ; test 9 711 normal (43 %) /
  12 833 attaques (57 %) dont ~17 types **absents du train** (mscan, apache2,
  saint, httptunnel, ps, xterm…).
- **Choix binaire** : `normal` → 0, tout le reste → 1. Simple, justifié pour
  un IDS (alerter sur toute intrusion), et il conserve les attaques rares.

## 2. Algorithmes et architecture

- **Prétraitement** : 3 catégorielles (`protocol_type`, `service`, `flag`) →
  OneHot (79 indicatrices, `handle_unknown="ignore"` pour les services inédits
  du test) ; 38 numériques → StandardScaler. Les deux ajustés **sur le train
  uniquement**, puis appliqués au test (pas de fuite).
- **SVM (RBF, défaut scikit-learn)** : classifieur supervisé sur 117 colonnes.
  Entraîné sur un **échantillon stratifié de 25 000 lignes** (seed 42) car un
  SVM RBF sur 125k lignes (~117 dims) est trop coûteux pour un TP (~50 s vs
  10–30 min estimées). Échantillonnage documenté et reproductible.
- **K-Means (k=2, `n_init=10`)** : clustering non supervisé sur les mêmes
  25 000 lignes, sans jamais voir les étiquettes.
- **Hybride (stacking, même méthode que phishing-detector)** : distances aux
  centres K-Means concaténées aux features, puis SVM mêmes hyperparamètres.
  Justification : réutilise l'info de regroupement sans fuite (K-Means fit sur
  train seul) et reste directement comparable au SVM seul (mêmes données,
  mêmes hyperparamètres — seul le vecteur d'entrée change).
- **Évaluation** : KDDTest+ **complet** (22 544 lignes), protocole officiel :
  mesure la généralisation aux attaques inédites, plus honnête qu'un split
  du train.

## 3. Expériences et résultats mesurés (sorties réelles de train.py)

| Modèle | Accuracy | Precision | Recall | F1 | FPR | Confusion [TN FP / FN TP] |
|---|---|---|---|---|---|---|
| SVM seul | 0.7934 | 0.9249 | 0.6934 | 0.7926 | 0.0743 | [[8989, 722], [3935, 8898]] |
| Hybride k=2 | 0.7622 | 0.9205 | 0.6372 | 0.7531 | 0.0727 | [[9005, 706], [4656, 8177]] |
| Hybride k=5 (variante) | 0.7600 | 0.9191 | 0.6341 | 0.7505 | 0.0737 | [[8995, 716], [4695, 8138]] |

Durée d'entraînement : ~45–55 s (échantillon 25k). Variante k=5 conservée
dans `variants/k5/`. Tests : 13/13 passent (`tests/test_pipeline.py`).

## 4. Analyse : pourquoi l'hybride ne gagne pas ici

Contrairement au projet phishing (hybride +0,2 pt), ici le stacking **perd**
3,1 pts d'accuracy et 5,6 pts de rappel. Explication : le test contient des
attaques inédites dont la « distance au cluster normal » ressemble à du
trafic normal — les 2 features ajoutées attirent la frontière SVM du mauvais
côté (721 FN de plus). Le FPR baisse légèrement (16 FP évités) : le
regroupement aide un peu côté normal, mais pas côté attaques inconnues.
La variante k=5 ne corrige pas le problème (même mécanisme). C'est un
**résultat honnête et défendable** : l'hybride stacking aide quand les
clusters séparent bien les classes (phishing), pas quand le test apporte des
familles inconnues (NSL-KDD).

## 5. Limites et pistes

- Dataset de 1999/2009 : ne représente pas le trafic moderne (pas de TLS 1.3,
  IoT, etc.) ; aucune capture temps réel — analyse de laboratoire uniquement.
- Échantillon 25k : reproductible mais sous-emploie le train ; un SVM linéaire
  sur 125k ou un MLP seraient des comparaisons naturelles.
- Binaire : on ne distingue pas les familles d'attaques (DoS/Probe/R2L/U2R).
- Pistes jury : règle OU (SVM=attaque OU loin du cluster normal → attaque,
  +recall/−FPR à quantifier), IsolationForest sur le résidu K-Means,
  calibration des scores, analyse d'erreurs par famille d'attaque du test.

## 6. Erreurs rencontrées et corrections

- Aucune donnée inventée : labels et répartitions comptés directement dans
  les `.txt` ; hyperparamètres et seed tracés dans `models/meta.json`.
- `OneHotEncoder(sparse_output=...)` : repli prévu pour scikit-learn < 1.2.
- Services inédits du test (ex. `mscan`-liés) : absorbés par
  `handle_unknown="ignore"` au lieu de planter.

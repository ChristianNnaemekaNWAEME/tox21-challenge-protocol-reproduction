# tox21-challenge-protocol-reproduction

A Tox21 Data Challenge-style evaluation: 12 assays, ROC-AUC per assay, mean over assays, computed with RDKit features and three standard models. It also measures how much a random split flatters the models compared with a scaffold split.

## Important limitation: not the original challenge data
The original 2014 Tox21 Data Challenge training/test files are not available to this study. The MoleculeNet copy of Tox21 (`tox21.csv`, 7,831 compounds) was used instead; a 2025 reproducible-leaderboard paper (arXiv:2511.14744) notes that such derived versions differ from the original challenge data. **Scores here are therefore not comparable with the published challenge leaderboard.**

## Method
- Features: 217 RDKit descriptors + 512-bin Morgan count fingerprint (about 730 numbers per molecule).
- Splits per assay, three seeds: (a) stratified random 80/20 (challenge-style) and (b) scaffold split (test molecules have unseen core structures).
- Models: LASSO (L1 logistic, penalty by 3-fold CV on training data), Random Forest (200 trees), histogram gradient boosting. Random Forest and boosting are untuned.

## Results (mean ROC-AUC, 12 assays x 3 seeds)
| model | random split | scaffold split | difference |
|---|---|---|---|
| Random Forest | 0.850 | 0.825 | 0.025 |
| Gradient boosting | 0.839 | 0.813 | 0.026 |
| LASSO | 0.808 | 0.779 | 0.029 |

A random split overstates accuracy on new chemistry by about 0.025 to 0.03 AUC for all three models, and the ranking of models does not change. Per-assay values are in `results/summary.txt`.

## Reproduce
```
pip install rdkit scikit-learn numpy pandas
# download tox21.csv from raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/tox21.csv
python -c "import numpy as np, pandas as pd; from lib import featurize; df=pd.read_csv('tox21.csv'); X,k,s=featurize(df.smiles.values); np.save('tox21_X.npy',X); np.save('tox21_keep.npy',k); np.save('tox21_scaf.npy',s)"
python challenge.py
```

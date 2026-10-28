"""Tox21 Challenge-style evaluation on the public MoleculeNet copy: per-assay ROC-AUC and the mean over 12 assays,
with a random (challenge-style) split and a scaffold split, to show how much a random split flatters models."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lib import scaffold_split, prep, lasso_logit
warnings.filterwarnings('ignore')
df=pd.read_csv('tox21.csv'); X=np.load('tox21_X.npy'); keep=np.load('tox21_keep.npy'); sc=np.load('tox21_scaf.npy'); d=df.iloc[keep].reset_index(drop=True)
tasks=[c for c in df.columns if c.startswith(('NR-','SR-'))]; rows=[]
for t in tasks:
    y=d[t].values; ok=~np.isnan(y); Xo=X[ok]; yo=y[ok].astype(int); so=sc[ok]
    for seed in range(3):
        splits={'random':train_test_split(np.arange(len(yo)),test_size=0.2,stratify=yo,random_state=seed),'scaffold':scaffold_split(so,seed)}
        for sp,(tr,te) in splits.items():
            Xtr,Xte=prep(Xo[tr].copy(),Xo[te].copy())
            for name,f in {'LASSO':lambda:lasso_logit(Xtr,yo[tr],seed),
                           'Random Forest':lambda:RandomForestClassifier(200,n_jobs=2,random_state=seed,class_weight='balanced_subsample',min_samples_leaf=2).fit(Xtr,yo[tr]),
                           'Gradient boosting':lambda:HistGradientBoostingClassifier(max_iter=150,random_state=seed).fit(Xtr,yo[tr])}.items():
                m=f(); p=m.decision_function(Xte) if hasattr(m,'decision_function') else m.predict_proba(Xte)[:,1]
                rows.append(dict(assay=t,seed=seed,split=sp,model=name,auc=roc_auc_score(yo[te],p)))
r=pd.DataFrame(rows); r.to_csv('results_tox21_challenge_protocol.csv',index=False)
m=r.groupby(['model','split']).auc.mean().unstack().round(3); m['random_minus_scaffold']=(m['random']-m['scaffold']).round(3); print(m)
print(r[r.split=='random'].groupby(['assay','model']).auc.mean().unstack().round(3))

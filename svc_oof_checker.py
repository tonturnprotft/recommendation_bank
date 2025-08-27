# svc_oof_checker.py
import os, numpy as np, pandas as pd

BASE = "/Users/tree/Projects/recommemdation_bank/outputs"
PRED = f"{BASE}/predictions/svc_rbf/svc_cv_preds.parquet"
LABELS_PAR = f"{BASE}/balanced/labels_mm_folded.parquet"  # set to None to skip ranking

def ndcg_at_k(y, s, k):
    k = min(k, len(y))
    if k <= 0: return np.nan
    order = np.argsort(-s)
    rel = y[order][:k]
    disc = 1.0 / np.log2(np.arange(2, k+2))
    dcg = np.sum((2**rel - 1) * disc)
    ideal = np.sort(y)[::-1][:k]
    idcg = np.sum((2**ideal - 1) * disc)
    return np.nan if idcg == 0 else dcg / idcg

def hit_at_k(y, s, k):
    if y.sum() == 0: return np.nan
    return 1.0 if y[np.argsort(-s)[:k]].sum() > 0 else 0.0

svc = pd.read_parquet(PRED)
print("SVC OOF shape:", svc.shape)
print("Columns:", list(svc.columns))
prob_cols = [c for c in svc.columns if c.endswith("_prob")]
print("Prob columns:", prob_cols)

print("NaN present?:", svc[prob_cols].isna().any().any())
print("Min/Max per prob col:\n", svc[prob_cols].agg(["min","max"]).T)

dups = svc["client_id"].duplicated().sum()
print("Duplicate client_ids:", dups)
print("Fold counts:", svc["fold"].value_counts().sort_index().to_dict())

if LABELS_PAR and os.path.exists(LABELS_PAR):
    labels = pd.read_parquet(LABELS_PAR)
    labels["client_id"] = labels["client_id"].astype(str)
    targets = [c for c in labels.columns if c.startswith("target_")]
    df = svc.merge(labels[["client_id","fold"]+targets], on=["client_id","fold"], how="left")

    rows = []
    for f in sorted(df["fold"].unique()):
        d = df[df.fold == f]
        Y = d[targets].values.astype(int)
        S = np.column_stack([d[f"{t}_prob"].values for t in targets])
        hits1, hits3, ndcg3, ndcg5 = [], [], [], []
        for i in range(len(d)):
            if Y[i].sum() == 0: 
                continue
            hits1.append(hit_at_k(Y[i], S[i], 1))
            hits3.append(hit_at_k(Y[i], S[i], 3))
            ndcg3.append(ndcg_at_k(Y[i], S[i], 3))
            ndcg5.append(ndcg_at_k(Y[i], S[i], 5))
        rows.append(dict(fold=int(f), users_eval=len(hits1),
                         Hit1=np.nanmean(hits1), Hit3=np.nanmean(hits3),
                         NDCG3=np.nanmean(ndcg3), NDCG5=np.nanmean(ndcg5)))
    res = pd.DataFrame(rows)
    print("\nRanking (avg over folds):")
    print(res.agg({"users_eval":"sum","Hit1":"mean","Hit3":"mean","NDCG3":"mean","NDCG5":"mean"}))
else:
    print("\n(labels file not found) Skipping ranking computation.")
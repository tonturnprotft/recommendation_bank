# 06_export_production_blended.py
# Train final LGBM + SVC on all labeled data, save artifacts for production inference.

import os, json, gc, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

warnings.filterwarnings("ignore")

# ====== CONFIG ======
BASE_OUT     = "/Users/tree/Projects/recommemdation_bank/outputs"
MM_JSONL     = f"{BASE_OUT}/json/mm/json_balanced_mm.jsonl"         # training texts (balanced)
LABELS_PAR   = f"{BASE_OUT}/balanced/labels_mm_folded.parquet"      # labels for those users

ARTI_DIR     = f"{BASE_OUT}/artifacts/blended"
MODELS_DIR   = os.path.join(ARTI_DIR, "models")
VECT_DIR     = os.path.join(ARTI_DIR, "vectors")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VECT_DIR, exist_ok=True)

# Blending weights from your grid search
W_LGBM = 0.90
W_SVC  = 0.10

# Feature pipeline (same as earlier runs) — NOTE: no custom preprocessor here
WORD_VECT = dict(
    max_features=200_000,
    ngram_range=(1,2),
    min_df=2, max_df=0.995,
    lowercase=True,
    sublinear_tf=True
)
CHAR_VECT = dict(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=2, max_df=1.0,
    lowercase=True
)

# SVC projection
SVD_DIMS = 512
RANDOM_STATE = 42

# LGBM params
try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit(
        f"LightGBM not installed: {e}\n"
        "Install with one of:\n"
        "  pip install lightgbm\n"
        "  conda install -c conda-forge lightgbm"
    )

LGBM_KW = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

def load_mm_texts(path):
    rows=[]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append((str(r["client_id"]), r["text"]))
    return pd.DataFrame(rows, columns=["client_id","text"])

def class_weights(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0: return None
    w_pos = neg / max(pos, 1)
    return np.where(y == 1, w_pos, 1.0)

# ====== Load data ======
texts  = load_mm_texts(MM_JSONL)
labels = pd.read_parquet(LABELS_PAR)
labels["client_id"] = labels["client_id"].astype(str)

df = labels.merge(texts, on="client_id", how="inner")
target_cols = [c for c in df.columns if c.startswith("target_")]
assert target_cols, "No target_* columns found."

print(f"Training on {len(df)} users with targets: {target_cols}")

# ====== Normalize text ONCE (replace '.' -> '_' to preserve tokens like a6.24) ======
df["text_norm"] = df["text"].str.replace('.', '_', regex=False)

# ====== Fit shared vectorizers on ALL normalized texts ======
vec_w = TfidfVectorizer(**WORD_VECT)
vec_c = TfidfVectorizer(**CHAR_VECT)
X_w = vec_w.fit_transform(df["text_norm"])
X_c = vec_c.fit_transform(df["text_norm"])
X_sparse = hstack([X_w, X_c], format="csr")

# For SVC: fit SVD + scaler on ALL texts
svd = TruncatedSVD(n_components=SVD_DIMS, random_state=RANDOM_STATE)
X_dense_svd = svd.fit_transform(X_sparse)
scaler = StandardScaler(with_mean=True)
X_dense_svd = scaler.fit_transform(X_dense_svd)

# ====== Train per-target models ======
PATH_LGBM = os.path.join(MODELS_DIR, "lgbm")
PATH_SVC  = os.path.join(MODELS_DIR, "svc_rbf")
os.makedirs(PATH_LGBM, exist_ok=True)
os.makedirs(PATH_SVC,  exist_ok=True)

model_index = {"targets": target_cols, "lgbm": {}, "svc": {}}

for t in target_cols:
    y = df[t].values.astype(int)
    w = class_weights(y)

    # LGBM (on sparse)
    lgbm = lgb.LGBMClassifier(**LGBM_KW, verbosity=-1)
    try:
        lgbm.fit(X_sparse, y, sample_weight=w)
    except TypeError:
        lgbm.fit(X_sparse, y)
    joblib.dump(lgbm, os.path.join(PATH_LGBM, f"{t}.pkl"))
    model_index["lgbm"][t] = f"models/lgbm/{t}.pkl"

    # SVC (on dense-svd)
    svc = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    try:
        svc.fit(X_dense_svd, y, sample_weight=w)
    except TypeError:
        svc.fit(X_dense_svd, y)
    joblib.dump(svc, os.path.join(PATH_SVC, f"{t}.pkl"))
    model_index["svc"][t] = f"models/svc_rbf/{t}.pkl"

    print(f"Trained & saved: {t}")

# ====== Save vectorizers / projectors / metadata ======
joblib.dump(vec_w,   os.path.join(VECT_DIR, "tfidf_word.pkl"))
joblib.dump(vec_c,   os.path.join(VECT_DIR, "tfidf_char.pkl"))
joblib.dump(svd,     os.path.join(VECT_DIR, f"svd_{SVD_DIMS}.pkl"))
joblib.dump(scaler,  os.path.join(VECT_DIR, "scaler.pkl"))

metadata = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "targets": target_cols,
    "blend_weights": {"lgbm": 0.90, "svc": 0.10},
    "vectorizers": {
        "tfidf_word": "vectors/tfidf_word.pkl",
        "tfidf_char": "vectors/tfidf_char.pkl",
        "svd":        f"vectors/svd_{SVD_DIMS}.pkl",
        "scaler":     "vectors/scaler.pkl"
    },
    "models": {"lgbm": {t: f"models/lgbm/{t}.pkl" for t in target_cols},
               "svc":  {t: f"models/svc_rbf/{t}.pkl" for t in target_cols}},
    "notes": "Final blended production models trained on balanced MM set. Proba = 0.90*LGBM + 0.10*SVC. Text normalized with '.'→'_'."
}
with open(os.path.join(ARTI_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== Artifacts saved ===")
print(ARTI_DIR)
print(json.dumps(metadata, indent=2))
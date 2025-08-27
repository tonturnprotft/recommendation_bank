# predict_blended.py
# Score new MM JSONL (client_id, text) using saved LGBM+SVC artifacts and output Top-K recs.

import os, json, argparse
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack

def load_jsonl(path):
    rows=[]
    with open(path) as f:
        for line in f:
            r=json.loads(line)
            rows.append((str(r["client_id"]), r["text"]))
    return pd.DataFrame(rows, columns=["client_id","text"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=False,
                    default="/Users/tree/Projects/recommemdation_bank/outputs/artifacts/blended",
                    help="Artifacts directory produced by 06_export_production_blended.py")
    ap.add_argument("--input", required=True, help="Input MM JSONL (client_id, text)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--topk", type=int, default=3, help="Top-K labels to return per user")
    args = ap.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    arti = args.artifacts
    meta_path = os.path.join(arti, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {arti}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    targets = meta["targets"]
    w_lgbm  = float(meta["blend_weights"]["lgbm"])
    w_svc   = float(meta["blend_weights"]["svc"])

    # Load vectorizers / projectors
    tfidf_word = joblib.load(os.path.join(arti, meta["vectorizers"]["tfidf_word"]))
    tfidf_char = joblib.load(os.path.join(arti, meta["vectorizers"]["tfidf_char"]))
    svd        = joblib.load(os.path.join(arti, meta["vectorizers"]["svd"]))
    scaler     = joblib.load(os.path.join(arti, meta["vectorizers"]["scaler"]))

    # Load models
    mdl_lgbm = {t: joblib.load(os.path.join(arti, meta["models"]["lgbm"][t])) for t in targets}
    mdl_svc  = {t: joblib.load(os.path.join(arti, meta["models"]["svc"][t]))  for t in targets}

    # Read & normalize input texts
    df = load_jsonl(args.input)
    if df.empty:
        raise SystemExit("No rows in input JSONL.")
    df["text_norm"] = df["text"].str.replace('.', '_', regex=False)

    # Build features ONCE
    X_w = tfidf_word.transform(df["text_norm"])
    X_c = tfidf_char.transform(df["text_norm"])
    X_sparse = hstack([X_w, X_c], format="csr")

    X_svd = svd.transform(X_sparse)
    X_svd = scaler.transform(X_svd)

    # Predict per target & blend
    probs = np.zeros((len(df), len(targets)), dtype=float)
    for ti, t in enumerate(targets):
        p_lgbm = mdl_lgbm[t].predict_proba(X_sparse)[:, 1]
        p_svc  = mdl_svc[t].predict_proba(X_svd)[:, 1]
        probs[:, ti] = w_lgbm * p_lgbm + w_svc * p_svc

    # Write JSONL: client_id, per-target probs, topK labels
    with open(args.out, "w") as fout:
        for i in range(len(df)):
            row = {"client_id": df.loc[i, "client_id"]}
            for ti, t in enumerate(targets):
                row[f"{t}_prob"] = float(probs[i, ti])
            order = np.argsort(-probs[i])
            topk = [targets[j] for j in order[:max(1, args.topk)]]
            row["topk"] = topk
            fout.write(json.dumps(row) + "\n")

    print(f"Saved predictions to: {args.out}")

if __name__ == "__main__":
    main()
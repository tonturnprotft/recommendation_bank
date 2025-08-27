import json, numpy as np, pandas as pd, os, itertools

RECS = "/Users/tree/Projects/recommemdation_bank/outputs/predictions/prod/recs.jsonl"

rows=[]
with open(RECS) as f:
    for i,line in enumerate(f):
        r=json.loads(line)
        rows.append(r)
        if i<5:  # show a peek
            print("SAMPLE:", r)

print("\nTotal rows:", len(rows))
df = pd.DataFrame(rows)

# infer target columns from *_prob keys
tcols = [c for c in df.columns if c.endswith("_prob")]
print("Targets (from file):", [c.replace("_prob","") for c in tcols])

# basic health checks
print("Any NaN probs?:", df[tcols].isna().any().any())
print("Prob ranges per target:")
print(df[tcols].agg(['min','max']).T)

# topk length check (every row should have list of len>=1)
bad_topk = df['topk'].map(lambda x: not isinstance(x, list) or len(x)==0).sum()
print("Rows with bad/empty topk:", bad_topk)

# Top-1 label distribution (which label wins most often)
def top1(row):
    probs = row[tcols].values
    j = int(np.argmax(probs))
    return tcols[j].replace("_prob","")
top1_counts = df.apply(top1, axis=1).value_counts()
print("\nTop-1 distribution:\n", top1_counts)

# quick uniqueness check
dups = df['client_id'].duplicated().sum()
print("Duplicate client_ids:", dups)
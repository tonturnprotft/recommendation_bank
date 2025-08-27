# recommendation_bank

Multi-label recommendation system that ranks **four targets (`target_1`…`target_4`) per user** from serialized transaction (TRX) and geolocation (GEO) histories.  
Best offline ranking uses a **weighted blend** of **LightGBM (90%) + SVC‑RBF (10%)** over TF‑IDF features.

- **Author:** Caio Appolinario, Dhinna Tretarnthip 
- **Date:** 2025-08-28

---

## 🚀 TL;DR

- Prepare per-user text corpora from TRX/GEO → JSONL.
- Train models (OOF CV) and compute blend tuned for ranking (NDCG@3).
- Export production artifacts (vectorizers + models + metadata).
- Score new users via `predict_blended.py` → per-label probabilities + Top‑K.

**Key OOF ranking (5-fold):**  
**Hit@1 = 0.5752**, **Hit@3 = 0.9809**, **NDCG@3 = 0.8124**, **NDCG@5 = 0.8214**

---

## 📁 Repository Layout

```
recommendation_bank/
├─ notebooks/
│  ├─ 00_config_and_balance.ipynb
│  ├─ 00a_env_and_targets_diagnose.ipynb
│  ├─ 01_trx_prepare_dataset.ipynb
│  ├─ 02_geo_prepare_dataset.ipynb
│  ├─ 03_trxgeo_prepare_dataset.ipynb
│  ├─ 04_check_data.ipynb
│  ├─ 05_train_baseline.ipynb
│  └─ 05_train_model_zoo.ipynb
├─ outputs/
│  ├─ artifacts/blended/           # exported prod artifacts (models, vecs, metadata.json)
│  ├─ balanced/                    # balanced ids & labels (parquet)
│  ├─ metrics/                     # CSV summaries
│  └─ predictions/                 # OOF/blend/prod predictions
├─ 06_export_production_blended.py # trains full-data LGBM+SVC + saves artifacts
├─ predict_blended.py              # CLI inference on new MM JSONL
├─ check_predict.py                # predictions sanity checks
└─ svc_oof_checker.py              # SVC OOF QC
```

> **Git ignore**: `data/` and `outputs/json/` are ignored to keep the repo light.  
> Track small artifacts (e.g., `outputs/artifacts/blended/metadata.json`, `outputs/metrics/**`, `outputs/predictions/prod/*`).  
> Consider **Git LFS** if you need to include `.parquet`/`.pkl` files.

---

## 📦 Environment

- Python **3.10–3.12**
- Recommended virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install numpy pandas scipy scikit-learn lightgbm joblib tqdm
pip install pyarrow fastparquet   # for Parquet I/O
```


---

## 🧱 Data Expectations

**Not in repo (ignored):**
- TRX parquet parts: `detail/trx/fold=*/part-*.parquet`
- GEO parquet parts: `detail/geo/fold=*/part-*.parquet`
- Targets parquet parts: `targets/fold=*/part-*.parquet` (must contain `client_id`, `target_1..target_4`, etc.)

Folds are `{0,1,2,3,4}`; we prevent leakage by grouping strictly by `client_id`.

---

## 🔧 Pipeline

1. **Build targets & balance** → `notebooks/00_config_and_balance.ipynb`  
   Outputs: `outputs/balanced/targets_raw.parquet`, `outputs/balanced/mbd_targets_balanced.parquet`

2. **TRX/GEO to JSONL** → `notebooks/01_*` and `02_*`  
   Outputs: `outputs/json/{trx|geo}/mbd_fold_*.jsonl` (+ balanced)

3. **Merge TRX+GEO → MM JSONL** → `notebooks/03_*`  
   Output: `outputs/json/mm/json_balanced_mm.jsonl`

4. **Checks** → `notebooks/04_check_data.ipynb`

5. **Train models (OOF)** → `notebooks/05_train_baseline.ipynb`, `05_train_model_zoo.ipynb`  
   - Produces OOF probabilities under `outputs/predictions/{lr|svc_rbf|lgbm}/`  
   - Metrics in `outputs/metrics/**`

6. **Export production blend** → `06_export_production_blended.py`  
   Writes to `outputs/artifacts/blended/`:
   - `vectors/` → TF‑IDF word/char, `svd_512.pkl`, `scaler.pkl`
   - `models/` → per‑target LGBM + SVC‑RBF
   - `metadata.json` → targets + blend weights (`w_lgbm=0.90`, `w_svc=0.10`)

---

## 📈 Results (OOF, 5 folds)

| Model                               | Hit@1 | Hit@3 | NDCG@3 | NDCG@5 |
|------------------------------------|:----:|:----:|:-----:|:-----:|
| Logistic Regression (TF‑IDF)       | 0.4678 | 0.9686 | 0.7590 | 0.7742 |
| SVC (RBF)                          | 0.5636 | 0.9800 | 0.8072 | 0.8166 |
| LightGBM                           | 0.5515 | 0.9745 | 0.7981 | 0.8096 |
| **Blended (0.90·LGBM + 0.10·SVC)** | **0.5752** | **0.9809** | **0.8124** | **0.8214** |

Per-target AUC/AP tables are in `outputs/metrics/**`.

---

## 🧪 Inference (CLI)

Input: MM JSONL (`client_id`, `text`). Output: per‑label probs + `topk`.

```bash
python3 predict_blended.py   --input outputs/json/mm/json_balanced_mm.jsonl   --out   outputs/predictions/prod/recs.jsonl   --topk  3
```

Optional convenience:
```bash
python3 check_predict.py  # quick QC
```

---


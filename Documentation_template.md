# Amazon ML Challenge 2026 - Business Entity Resolution
## Methodology Document

**Team Name:** [Your Team Name]
**Team Members:** [Member 1, Member 2, ...]
**Submission Date:** [Date]

---

## 1. Executive Summary

This document describes our approach to the Amazon ML Challenge 2026 Business Entity Resolution task. We developed a two-stage pipeline consisting of (1) a high-recall hybrid blocking strategy combining inverted indexing with embedding-based approximate nearest neighbors, and (2) an ensemble matching model using gradient-boosted trees (XGBoost, LightGBM, CatBoost) with cross-encoder re-ranking, optimized for the macro F0.5 metric which weights precision 2× over recall.

**Key Results:**
- Training Macro F0.5: [XX.XX%]
- Validation Macro F0.5: [XX.XX%]
- Blocking Recall (estimated): [XX.XX%]
- Inference time: [XX minutes on test set]

---

## 2. Problem Analysis

### 2.1 Challenge Characteristics
- **Three heterogeneous sources**: Source 1 (clean reference), Sources 2 & 3 (noisy vendor feeds)
- **No shared identifiers**: Only business name and address available
- **Cardinality**: One-to-many mapping from Source 1 to Sources 2/3
- **Singleton entities**: Source 1 entities with zero matches in Sources 2/3
- **Metric**: Macro F0.5 (precision-weighted) → Conservative merging strategy required

### 2.2 Key Challenges Addressed
1. **Scale**: Full cross-product O(N₁×(N₂+N₃)) infeasible → Blocking essential
2. **Noise**: Abbreviations, landmark references, formatting differences
3. **Geographic variation**: Address conventions differ by region/country
4. **Singleton detection**: Critical for F0.5 (FP on singleton = 0 score)

---

## 3. Stage 1: Blocking (Candidate Generation)

### 3.1 Design Philosophy
Blocking determines the recall ceiling. We prioritize **high recall** while keeping candidate sets manageable (~100-200 per Source 1 entity). Our hybrid approach combines:

| Strategy | Keys Generated | Strength |
|----------|---------------|----------|
| Token-based Inverted Index | Name tokens, address tokens, n-grams, phonetic codes | High recall, interpretable |
| Embedding-based ANN | MiniLM/MiniLM/BGE embeddings + HNSW | Semantic similarity, catches paraphrases |
| Cross-field Keys | Name token + postal prefix, first tokens | Joint name-address signals |

### 3.2 Blocking Key Details

#### 3.2.1 Name-Based Keys
- **First token**: `name_first:{token}` — captures primary business identifier
- **First two tokens**: `name_first2:{t1}_{t2}` — handles "Acme Robotics" vs "Acme Robotics Inc"
- **Individual tokens**: `name_token:{token}` — inverted index for token overlap
- **Character 3-grams**: `name_ngram:{ngram}` — robust to typos/abbreviations
- **Phonetic codes**: Soundex, Metaphone, NYSIIS — handles spelling variations

#### 3.2.2 Address-Based Keys
- **Postal code prefix** (3 digits): `postal_prefix:{prefix}` — geographic locality
- **Full postal code**: `postal_full:{code}` — exact area match
- **House number**: `house_num:{num}` — precise location
- **Street tokens**: `street_token:{token}` — street name matching
- **City/State**: `city:{city}`, `state:{state}` — administrative boundaries
- **Address n-grams**: `addr_ngram:{ngram}` — fuzzy address matching

#### 3.2.3 Cross-Field Keys
- **Name token + postal prefix**: `cross:{name_token}_{postal_prefix}`
- **First name token + first address token**: `cross_first:{name_token}_{addr_token}`

#### 3.2.4 Embedding-Based Blocking
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Text**: `"{name} [SEP] {address}"`
- **Index**: HNSW (ef_construction=200, M=16)
- **Neighbors**: Top-50 per query
- **Fallback**: sklearn NearestNeighbors if hnswlib unavailable

### 3.3 Candidate Budget Allocation
- **Inverted index**: 70% of budget (~140 candidates)
- **Embedding ANN**: 30% of budget (~60 candidates)
- **Deduplication**: Union of both strategies, capped at 200
- **Minimum guarantee**: At least 1 candidate per Source 1 entity

### 3.4 Blocking Quality Metrics
| Metric | Value |
|--------|-------|
| Unique blocking keys | [XX,XXX] |
| Avg candidates per Source 1 | [XXX] |
| Estimated recall (train) | [XX.XX%] |
| Singleton coverage | [XX.XX%] |

---

## 4. Stage 2: Matching Model (Pairwise Classification)

### 4.1 Feature Engineering

#### 4.1.1 Name Features (22 features)
| Category | Features |
|----------|----------|
| Exact/Prefix/Suffix | `exact_match`, `prefix_match`, `suffix_match` |
| Token Overlap | `token_jaccard`, `token_overlap_ratio`, `token_count_ratio`, `common_token_count`, `unique_token_count` |
| Abbreviation | `abbreviation_match` |
| Phonetic | `soundex_match`, `metaphone_match`, `nysiis_match` |
| String Metrics | `jaro_winkler`, `levenshtein_norm` |
| TF-IDF | `tfidf_cosine` (char 3-5 grams) |
| Embeddings | `emb_minilm_cosine`, `emb_mpnet_cosine`, `emb_bge_cosine` |

#### 4.1.2 Address Features (19 features)
| Category | Features |
|----------|----------|
| Exact | `exact_match` |
| Components | `house_num_match`, `street_jaccard`, `street_overlap`, `city_match`, `state_match`, `postal_match`, `postal_prefix_match` |
| Token Overlap | `token_jaccard`, `token_overlap`, `landmark_overlap` |
| String Metrics | `jaro_winkler`, `levenshtein_norm` |
| TF-IDF | `tfidf_cosine` (word 1-3 grams) |
| Embeddings | `emb_minilm_cosine`, `emb_mpnet_cosine`, `emb_bge_cosine` |

#### 4.1.3 Cross Features (4 features)
- `cross_name1_addr2_overlap`, `cross_name2_addr1_overlap`
- `cross_name_name_overlap`, `cross_addr_addr_overlap`
- `cross_name_addr_consistency`

**Total: 45 features per pair**

### 4.2 Model Architecture

#### 4.2.1 Base Models (Tree Ensembles)
| Model | Key Parameters | Weight |
|-------|----------------|--------|
| XGBoost | n_est=500, max_depth=8, lr=0.05, subsample=0.8 | 0.35 |
| LightGBM | n_est=500, max_depth=8, lr=0.05, num_leaves=63 | 0.35 |
| CatBoost | iter=500, depth=8, lr=0.05, l2_leaf_reg=3 | 0.20 |

**Training Details:**
- 5-fold Stratified CV for robust estimation
- Class weights balanced (positive class ~1-5%)
- Early stopping on validation fold
- Calibrated probabilities via Platt scaling

#### 4.2.2 Cross-Encoder Re-ranking (Weight: 0.10)
- **Models**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Input**: `"{name1} [SEP] {address1} [SEP] {name2} [SEP] {address2}"`
- **Application**: Re-rank top-50 candidates per entity before final thresholding
- **Fusion**: Weighted average with tree ensemble probabilities

#### 4.2.3 Ensemble Fusion
```
P_final = 0.35 × P_xgb + 0.35 × P_lgb + 0.20 × P_cat + 0.10 × P_cross_encoder
```

### 4.3 Threshold Optimization for F0.5

Given macro F0.5 weights precision 2× recall:
- **Search space**: 0.10 to 0.90 (81 steps)
- **Objective**: Maximize F0.5 on validation fold
- **Result**: Optimal threshold = [0.XX] (typically 0.65-0.85 for F0.5)
- **Singleton handling**: Entities with max probability < threshold → empty prediction

### 4.4 Handling Singletons
- **Detection**: If max candidate probability < `singleton_threshold` (tuned separately, ~0.3)
- **Action**: Predict empty match list
- **Validation**: Ensures singleton F1 = 1.0 when correct

---

## 5. Training & Validation Strategy

### 5.1 Data Splits
- **Train/Val**: 80/20 stratified by Source 1 entity
- **Cross-validation**: 5-fold for model selection and threshold tuning
- **No leakage**: Blocking keys derived only from training data

### 5.2 Class Imbalance Handling
- Positive pairs: ~1-5% of candidates
- **Techniques**: 
  - `scale_pos_weight` in XGBoost/LightGBM
  - Class weights in CatBoost
  - Focal loss exploration (not used in final)
  - Under-sampling negatives for cross-encoder fine-tuning

### 5.3 Hyperparameter Optimization
- **Method**: Optuna (50 trials) on validation F0.5
- **Key params tuned**: Learning rate, depth, regularization, ensemble weights
- **Result**: See Section 4.2.1 for final values

---

## 6. Computational Requirements

### 6.1 Training Time (Estimated)
| Stage | Time | Hardware |
|-------|------|----------|
| Blocking Index Build | ~10 min | CPU (16 cores) |
| TF-IDF Fitting | ~5 min | CPU |
| Feature Extraction | ~30 min | CPU |
| Model Training (3×GBDT) | ~45 min | CPU/GPU |
| Cross-Encoder (optional) | ~60 min | GPU |
| **Total** | **~2.5 hours** | |

### 6.2 Inference Time (Test Set)
| Stage | Time |
|-------|------|
| Blocking | ~5 min |
| Feature Extraction | ~15 min |
| Model Inference | ~10 min |
| **Total** | **~30 min** |

### 6.3 Memory Requirements
- Blocking index: ~2-4 GB
- Feature matrix: ~1-2 GB (sparse)
- Models: ~500 MB
- **Peak RAM**: ~8-12 GB

---

## 7. Region-Specific Normalization

### 7.1 Business Name Normalization
- Unicode NFKC normalization
- Accent removal
- Business suffix standardization (Inc/LLC/Corp/Co/etc.)
- Stopword removal for token-based matching

### 7.2 Address Normalization
- Abbreviation expansion (St→Street, Ave→Avenue, etc.)
- Directional standardization (N→North, SW→Southwest)
- Postal code extraction (US ZIP, CA, UK, generic)
- House number extraction
- Landmark token identification

### 7.3 Geographic Adaptation
- Postal code prefix blocking adapts to format
- Address component parsing handles comma-separated formats
- Phonetic algorithms work across Latin-script languages

---

## 8. Ablation Studies

| Configuration | Val F0.5 | Δ |
|---------------|----------|---|
| Full Ensemble (XGB+LGB+Cat+CE) | [XX.XX] | — |
| - Cross-Encoder | [XX.XX] | -X.XX |
| - CatBoost | [XX.XX] | -X.XX |
| - LightGBM | [XX.XX] | -X.XX |
| - XGBoost | [XX.XX] | -X.XX |
| Single Model (XGB only) | [XX.XX] | -X.XX |
| No Embedding Features | [XX.XX] | -X.XX |
| No Phonetic Features | [XX.XX] | -X.XX |
| Inverted Index Only (no ANN) | [XX.XX] | -X.XX |
| ANN Only (no Inverted) | [XX.XX] | -X.XX |

---

## 9. Error Analysis

### 9.1 False Positives (Precision Errors)
- **Similar names, different locations**: "Acme Robotics" at 123 Main St vs 456 Oak Ave
- **Franchise/chain confusion**: Same name, different franchisees
- **Abbreviation over-match**: "A&R Construction" vs "A&R Consulting"

### 9.2 False Negatives (Recall Errors)
- **Heavy abbreviation**: "Intl Biz Mach Corp" vs "International Business Machines"
- **Landmark-only addresses**: "Nr City Hall" vs "123 Main St"
- **Transliteration differences**: Non-Latin script variations
- **Missing from blocking**: True pairs not sharing any blocking key

### 9.3 Singleton Errors
- **Over-prediction**: Low-confidence match on true singleton
- **Under-prediction**: True singleton but candidate has moderate similarity

---

## 10. Reproducibility

### 10.1 Environment
```yaml
python: 3.10+
key_dependencies:
  - pandas, numpy, scikit-learn
  - xgboost, lightgbm, catboost
  - sentence-transformers, hnswlib
  - pyyaml, tqdm
```

### 10.2 Random Seeds
- All models: `random_state=42`
- Data splits: `random_state=42`
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

### 10.3 Run Instructions
```bash
# Training
python -m src.pipeline --mode train --data-dir data/train --model-dir models

# Inference
python -m src.pipeline --mode predict --data-dir data/test --model-dir models --output submission/matching_results.tsv

# Validation
python -m src.pipeline --mode validate --output submission/matching_results.tsv --ground-truth data/train/ground_truth_train.tsv
```

### 10.4 Expected Outputs
- `candidate_pairs.tsv` — Blocking output (audit)
- `matching_results.tsv` — Final predictions (leaderboard)
- `models/` — Trained models for reproduction

---

## 11. Future Improvements

1. **Active Learning**: Human-in-the-loop for borderline cases
2. **Graph-based Resolution**: Connected components over pairwise matches
3. **Multi-lingual Embeddings**: Better handling of non-English names
4. **Generative Matching**: LLM-based reasoning for ambiguous pairs
5. **Online Learning**: Incremental updates as new data arrives

---

## 12. References

- Fellegi, I. P., & Sunter, A. B. (1969). A theory for record linkage.
- Christen, P. (2012). Data Matching: Concepts and Techniques.
- Rekatsinas, T., et al. (2017). Socratic: End-to-end entity resolution.
- [Amazon ML Challenge 2026 Problem Statement]

---

*End of Document*